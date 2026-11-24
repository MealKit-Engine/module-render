#pragma once

#include "mealkit/render/DrawList.h"
#include "mealkit/render/Renderer.h"
#include "mealkit/core/Reflect.h"
#include "mealkit/scene/Scene.h"
#include "mealkit/sim/Table.h"

#include <string>
#include <string_view>
#include <unordered_map>

// Sprites in the world (docs/DECISIONS.md 2026-11-10 "2D games run on the 3D engine"): sprite images and the scene's
// sprite objects (scene::kTypeSprite). View only: nothing here touches the simulation.
namespace mealkit::render {

// PNG images (relative to assets) uploaded once as sprite textures: premultiplied alpha, sRGB, full mip chain (filtered in
// linear premultiplied space, so edges don't darken when scaled down). A file that changed on disk is reloaded.
class SpriteTextures {
public:
    struct Texture {
        u32 index = 0;          // bindless index (Sprite::texture)
        u32 width = 0, height = 0; // pixels (0 for missing files, which draw white)
    };
    explicit SpriteTextures(Renderer& renderer) : m_renderer(renderer) {}
    // Outside a frame or inside one before DrawList::submit. An empty path or a missing file gives the white texture
    const Texture& get(const std::string& assetPath);

private:
    struct Entry {
        Texture texture;
        i64 writeTime = 0;
    };
    Renderer& m_renderer;
    std::unordered_map<std::string, Entry> m_entries;
    Texture m_white;
};

// Flipbook clip (docs/DECISIONS.md 2026-11-23 "Flipbook clips are table data"): frames of a sheet played at a rate. One row of
// assets/tables/sprite_clips.table per clip, row id = clip name. Frames are cells firstCell .. firstCell + frames - 1, counted
// from the top left row by row (like SpriteFields::cell). View only: time is the view's clock, never the simulation tick.
MEALKIT_STRUCT(SpriteClip) {
    MEALKIT_FIELD(std::string, image, {}, "Sheet image (PNG relative to assets)");
    MEALKIT_FIELD(i32, columns, 1, "Sheet columns");
    MEALKIT_FIELD(i32, rows, 1, "Sheet rows");
    MEALKIT_FIELD(i32, firstCell, 0, "First cell");
    MEALKIT_FIELD(i32, frames, 1, "Frame count");
    MEALKIT_FIELD(f32, fps, 8.0f, "Frames per second");
    MEALKIT_FIELD(bool, loop, true, "Loop (off: stops on the last frame)");
};
inline constexpr const char* kSpriteClipTable = "tables/sprite_clips.table"; // relative to assets

class SpriteClips {
public:
    // Reads assets/tables/sprite_clips.table (false if missing: no clips)
    bool load();
    // Rereads the table when the file changed (checked every 0.25 s of dt). True if reloaded
    bool poll(f64 dt);
    const SpriteClip* find(std::string_view name) const; // nullptr if missing
    u32 count() const { return m_table.size(); }
    // Frame (0..frames-1) shown after `seconds` since the clip started (a negative time shows frame 0)
    static u32 frameAt(const SpriteClip& clip, f64 seconds);
    // Cell of the sheet shown after `seconds`
    static u32 cellAt(const SpriteClip& clip, f64 seconds);

private:
    sim::Table<SpriteClip> m_table;
    std::string m_path;
    unsigned long long m_time = 0;
    f64 m_check = 0.0;
};

// What game view code gives addSprite to play clips. clips == nullptr draws the object's own image and cell. clip (if not empty)
// replaces the object's `clip` field; start is the view time the clip began (restart a clip by passing a new start)
struct SpriteAnim {
    const SpriteClips* clips = nullptr;
    f64 time = 0.0;
    std::string_view clip;
    f64 start = 0.0;
};

// Pixel snapping for an orthographic 2D camera (docs/DECISIONS.md 2026-11-23 "Pixel snapping"). Makes every art pixel cover a
// whole number of screen pixels (zoom = screen pixels per art pixel, at least 1, rounded from the requested halfHeight) and puts the
// center on the screen-pixel grid so sprites don't shimmer while the camera moves. pixelsPerUnit <= 0 or a viewport without height
// returns the request unchanged. Sprite positions aren't touched. View only.
struct PixelView {
    Vec2 center;
    f32 halfHeight = 1.0f;
    u32 zoom = 1; // screen pixels per art pixel (0 when snapping is off)
};
PixelView snapPixelView(Vec2 center, f32 halfHeight, f32 viewportHeightPixels, f32 pixelsPerUnit);

// Fills a sprite for the image rect of one sheet cell (columns x rows, cell counted from the top left) and its pixel size
void spriteCell(Sprite& out, u32 columns, u32 rows, u32 cell);

// One scene sprite object (fields SpriteFields): world transform from the caller (flattened scene or worldTransform), the
// object's color as tint. flags is added to the sprite's (none defined for highlight yet)
void addSprite(DrawList& out, SpriteTextures& textures, const scene::Entity& e, const scene::Transform& world, const SpriteAnim* anim = nullptr);

// World rectangle the camera sees (x/y, meters); tilemaps skip cells outside it. View only
struct TileView {
    f32 minX = -1e30f, minY = -1e30f, maxX = 1e30f, maxY = 1e30f;
};

// One tilemap scene object (fields TilemapFields; docs/DECISIONS.md 2026-11-24 "Tilemaps are one scene object"): every non-empty cell becomes
// a sprite of the sheet cell, all with one texture so they batch into one or two draws. Cells outside `view` (if given) are skipped, which
// keeps a big map under kMaxSpritesPerFrame. world.position is the grid's bottom left corner (rotation and scale are ignored, like the collision);
// the object's color tints every cell
void addTilemap(DrawList& out, SpriteTextures& textures, const scene::Entity& e, const scene::Transform& world, const TileView* view = nullptr);

} // namespace mealkit::render
