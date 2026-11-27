#include "mealkit/render/Sprites.h"

#include "mealkit/core/Log.h"
#include "mealkit/data/Tables.h"
#include "mealkit/platform/Paths.h"

#include <algorithm>
#include <cmath>
#include <vector>

// Sprite PNGs are decoded in shipping builds too (unlike model textures, which ship cooked). Static copy so it doesn't
// clash with the other stb_image copies (ui/StbImpl.cpp, asset/Terrain.cpp)
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "stb_image.h"

namespace mealkit::render {
namespace {

f32 toLinear(u8 v) {
    const f32 c = static_cast<f32>(v) / 255.0f;
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

u8 toSrgb(f32 c) {
    c = std::clamp(c, 0.0f, 1.0f);
    const f32 s = c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    return static_cast<u8>(std::lround(s * 255.0f));
}

// RGBA8 straight sRGB → mip chain of premultiplied linear-filtered pixels, encoded back to sRGB (alpha stays linear)
// The size a PNG had before the browser cook halved it (game/Cook.h): chunk "mkSz" before IDAT, big-endian u32 width and height.
// false without one
bool originalSize(const std::string& png, u32& width, u32& height) {
    const auto be32 = [&](size_t at) {
        return static_cast<u32>(static_cast<u8>(png[at])) << 24 | static_cast<u32>(static_cast<u8>(png[at + 1])) << 16 |
               static_cast<u32>(static_cast<u8>(png[at + 2])) << 8 | static_cast<u32>(static_cast<u8>(png[at + 3]));
    };
    for (size_t at = 8; at + 12 <= png.size();) {
        const u32 length = be32(at);
        if (png.compare(at + 4, 4, "IDAT") == 0 || at + 12 + static_cast<size_t>(length) > png.size()) return false;
        if (length == 8 && png.compare(at + 4, 4, "mkSz") == 0) {
            const u32 w = be32(at + 8), h = be32(at + 12);
            if (!w || !h) return false;
            width = w;
            height = h;
            return true;
        }
        at += 12 + static_cast<size_t>(length);
    }
    return false;
}

std::vector<u8> buildMips(const u8* rgba, u32 width, u32 height, u32& mipCount) {
    std::vector<f32> level(static_cast<size_t>(width) * height * 4);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
        const f32 a = static_cast<f32>(rgba[i * 4 + 3]) / 255.0f;
        for (int c = 0; c < 3; ++c) level[i * 4 + c] = toLinear(rgba[i * 4 + c]) * a;
        level[i * 4 + 3] = a;
    }
    std::vector<u8> out;
    mipCount = 0;
    u32 w = width, h = height;
    while (true) {
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
            for (int c = 0; c < 3; ++c) out.push_back(toSrgb(level[i * 4 + c]));
            out.push_back(static_cast<u8>(std::lround(std::clamp(level[i * 4 + 3], 0.0f, 1.0f) * 255.0f)));
        }
        ++mipCount;
        if (w == 1 && h == 1) break;
        const u32 nw = std::max(1u, w / 2), nh = std::max(1u, h / 2);
        std::vector<f32> next(static_cast<size_t>(nw) * nh * 4);
        for (u32 y = 0; y < nh; ++y)
            for (u32 x = 0; x < nw; ++x)
                for (int c = 0; c < 4; ++c) { // 2x2 box (an odd last row/column is folded into the clamp)
                    const u32 x0 = std::min(x * 2, w - 1), x1 = std::min(x * 2 + 1, w - 1);
                    const u32 y0 = std::min(y * 2, h - 1), y1 = std::min(y * 2 + 1, h - 1);
                    const auto at = [&](u32 px, u32 py) { return level[(static_cast<size_t>(py) * w + px) * 4 + c]; };
                    next[(static_cast<size_t>(y) * nw + x) * 4 + c] = (at(x0, y0) + at(x1, y0) + at(x0, y1) + at(x1, y1)) * 0.25f;
                }
        level.swap(next);
        w = nw;
        h = nh;
    }
    return out;
}

} // namespace

const SpriteTextures::Texture& SpriteTextures::get(const std::string& assetPath) {
    if (m_white.index == 0) {
        const u8 white[4] = {255, 255, 255, 255};
        m_white.index = m_renderer.createTexture({1, 1, 1, rhi::Format::Rgba8Srgb, white});
    }
    if (assetPath.empty()) return m_white;
    const std::string file = findAssetFile(assetPath);
    const i64 time = file.empty() ? 0 : static_cast<i64>(fileWriteTime(file));
    auto it = m_entries.find(assetPath);
    if (it != m_entries.end() && it->second.writeTime == time) return it->second.texture.width ? it->second.texture : m_white;
    Entry& entry = m_entries[assetPath];
    entry.writeTime = time; // a broken or missing file is tried again only when it changes
    std::string bytes;
    if (file.empty() || !readTextFile(file, bytes)) {
        MEALKIT_WARN("Sprite: %s not found", assetPath.c_str());
        entry.texture = {};
        return m_white;
    }
    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()), &w, &h, &channels, 4);
    if (!pixels) {
        MEALKIT_WARN("Sprite: cannot decode %s (%s)", assetPath.c_str(), stbi_failure_reason());
        entry.texture = {};
        return m_white;
    }
    u32 mips = 0;
    const std::vector<u8> data = buildMips(pixels, static_cast<u32>(w), static_cast<u32>(h), mips);
    stbi_image_free(pixels);
    // A changed file gets a new texture; the old one stays in the bindless heap (the renderer doesn't delete textures)
    entry.texture.index = m_renderer.createTexture({static_cast<u32>(w), static_cast<u32>(h), mips, rhi::Format::Rgba8Srgb, data.data()});
    entry.texture.width = static_cast<u32>(w);
    entry.texture.height = static_cast<u32>(h);
    originalSize(bytes, entry.texture.width, entry.texture.height);
    return entry.texture;
}

bool SpriteClips::load() {
    m_path = findAsset(std::string("assets/") + kSpriteClipTable);
    m_time = m_path.empty() ? 0 : fileWriteTime(m_path);
    m_table = {};
    if (!m_path.empty()) data::loadTable(m_table, kSpriteClipTable);
    return !m_path.empty();
}

bool SpriteClips::poll(f64 dt) {
    m_check -= dt;
    if (m_check > 0.0) return false;
    m_check = 0.25;
    const std::string path = findAsset(std::string("assets/") + kSpriteClipTable);
    const unsigned long long t = path.empty() ? 0 : fileWriteTime(path);
    if (path == m_path && t == m_time) return false;
    load();
    MEALKIT_INFO("Sprites: %u clip(s) from %s", count(), path.empty() ? "(no sprite_clips.table)" : path.c_str());
    return true;
}

const SpriteClip* SpriteClips::find(std::string_view name) const {
    const u32 row = m_table.find(name);
    return row == kNoRow ? nullptr : &m_table[row];
}

u32 SpriteClips::frameAt(const SpriteClip& clip, f64 seconds) {
    const i64 frames = std::max<i64>(clip.frames, 1);
    if (!(seconds > 0.0) || !(clip.fps > 0.0f)) return 0;
    const i64 step = static_cast<i64>(std::floor(seconds * static_cast<f64>(clip.fps)));
    return static_cast<u32>(clip.loop ? step % frames : std::min(step, frames - 1));
}

u32 SpriteClips::cellAt(const SpriteClip& clip, f64 seconds) {
    return static_cast<u32>(std::max(clip.firstCell, 0)) + frameAt(clip, seconds);
}

PixelView snapPixelView(Vec2 center, f32 halfHeight, f32 viewportHeightPixels, f32 pixelsPerUnit) {
    PixelView out{center, halfHeight, 0};
    if (!(pixelsPerUnit > 0.0f) || !(viewportHeightPixels >= 1.0f) || !(halfHeight > 0.0f)) return out;
    const f32 requested = viewportHeightPixels / (2.0f * halfHeight * pixelsPerUnit); // screen pixels per art pixel as asked
    const f32 zoom = std::max(1.0f, std::floor(requested + 0.5f));
    out.zoom = static_cast<u32>(zoom);
    out.halfHeight = viewportHeightPixels / (2.0f * pixelsPerUnit * zoom);
    const f32 step = 1.0f / (pixelsPerUnit * zoom); // one screen pixel in meters
    out.center.x = std::floor(center.x / step + 0.5f) * step;
    out.center.y = std::floor(center.y / step + 0.5f) * step;
    return out;
}

void spriteCell(Sprite& out, u32 columns, u32 rows, u32 cell) {
    columns = std::max(columns, 1u);
    rows = std::max(rows, 1u);
    cell = std::min(cell, columns * rows - 1);
    const f32 cw = 1.0f / static_cast<f32>(columns), ch = 1.0f / static_cast<f32>(rows);
    out.u0 = static_cast<f32>(cell % columns) * cw;
    out.v0 = static_cast<f32>(cell / columns) * ch;
    out.u1 = out.u0 + cw;
    out.v1 = out.v0 + ch;
}

void addSprite(DrawList& out, SpriteTextures& textures, const scene::Entity& e, const scene::Transform& world, const SpriteAnim* anim) {
    scene::SpriteFields f = scene::readObject<scene::SpriteFields>(e);
    if (anim && anim->clips) { // a clip (the view's choice, else the object's) replaces image, sheet and cell; an unknown name keeps the object's own
        const std::string_view name = anim->clip.empty() ? std::string_view(f.clip) : anim->clip;
        if (const SpriteClip* clip = name.empty() ? nullptr : anim->clips->find(name)) {
            f.image = clip->image;
            f.columns = clip->columns;
            f.rows = clip->rows;
            f.cell = static_cast<i32>(SpriteClips::cellAt(*clip, anim->time - anim->start));
        }
    }
    const SpriteTextures::Texture& tex = textures.get(f.image);
    Sprite s;
    s.position = world.position;
    s.rotation = world.rotation;
    s.texture = tex.index;
    const u32 columns = static_cast<u32>(std::max(f.columns, 1)), rows = static_cast<u32>(std::max(f.rows, 1));
    spriteCell(s, columns, rows, static_cast<u32>(std::max(f.cell, 0)));
    if (f.pixelsPerUnit > 0.0f && tex.width) {
        s.width = static_cast<f32>(tex.width) / static_cast<f32>(columns) / f.pixelsPerUnit * world.scale.x;
        s.height = static_cast<f32>(tex.height) / static_cast<f32>(rows) / f.pixelsPerUnit * world.scale.y;
    } else {
        s.width = world.scale.x;
        s.height = world.scale.y;
    }
    s.pivotX = f.pivotX;
    s.pivotY = f.pivotY;
    s.color = e.color;
    s.layer = f.layer;
    s.flags = (f.flipX ? kSpriteFlipX : 0u) | (f.flipY ? kSpriteFlipY : 0u) | (f.pixelArt ? kSpritePoint : 0u);
    out.addSprite(s);
}

void addTilemap(DrawList& out, SpriteTextures& textures, const scene::Entity& e, const scene::Transform& world, const TileView* view) {
    const scene::TilemapFields f = scene::readObject<scene::TilemapFields>(e);
    const std::vector<i32> tiles = scene::parseTileCells(f.cells, f.width, f.height);
    if (f.width <= 0 || tiles.empty() || !(f.cellSize > 0.0f)) return;
    const i32 width = f.width, height = static_cast<i32>(tiles.size() / static_cast<size_t>(width));
    const SpriteTextures::Texture& tex = textures.get(f.image);
    const u32 columns = static_cast<u32>(std::max(f.columns, 1)), rows = static_cast<u32>(std::max(f.rows, 1));
    // Cell ranges the view touches
    i32 c0 = 0, c1 = width - 1, r0 = 0, r1 = height - 1;
    if (view) {
        const f32 cs = f.cellSize;
        c0 = std::max(c0, static_cast<i32>(std::floor((view->minX - world.position.x) / cs)));
        c1 = std::min(c1, static_cast<i32>(std::floor((view->maxX - world.position.x) / cs)));
        r0 = std::max(r0, static_cast<i32>(std::floor(static_cast<f32>(height) - (view->maxY - world.position.y) / cs)));
        r1 = std::min(r1, static_cast<i32>(std::floor(static_cast<f32>(height) - (view->minY - world.position.y) / cs)));
    }
    for (i32 r = r0; r <= r1; ++r) {
        for (i32 c = c0; c <= c1; ++c) {
            const i32 tile = tiles[static_cast<size_t>(r) * static_cast<size_t>(width) + static_cast<size_t>(c)];
            if (tile < 0) continue;
            Sprite s;
            s.position = {world.position.x + static_cast<f32>(c) * f.cellSize, world.position.y + static_cast<f32>(height - 1 - r) * f.cellSize, world.position.z};
            s.texture = tex.index;
            spriteCell(s, columns, rows, static_cast<u32>(tile));
            s.width = f.cellSize;
            s.height = f.cellSize;
            s.pivotX = 0.0f; // the cell's bottom left corner
            s.pivotY = 0.0f;
            s.color = e.color;
            s.layer = f.layer;
            s.flags = f.pixelArt ? kSpritePoint : 0u;
            out.addSprite(s);
        }
    }
}

} // namespace mealkit::render
