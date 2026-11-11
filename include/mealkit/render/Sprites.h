#pragma once

#include "mealkit/render/DrawList.h"
#include "mealkit/render/Renderer.h"
#include "mealkit/scene/Scene.h"

#include <string>
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

// Fills a sprite for the image rect of one sheet cell (columns x rows, cell counted from the top left) and its pixel size
void spriteCell(Sprite& out, u32 columns, u32 rows, u32 cell);

// One scene sprite object (fields SpriteFields): world transform from the caller (flattened scene or worldTransform), the
// object's color as tint. flags is added to the sprite's (none defined for highlight yet)
void addSprite(DrawList& out, SpriteTextures& textures, const scene::Entity& e, const scene::Transform& world);

} // namespace mealkit::render
