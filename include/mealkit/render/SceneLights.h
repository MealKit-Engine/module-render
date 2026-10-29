#pragma once

#include "mealkit/render/Renderer.h"
#include "mealkit/scene/Scene.h"

#include <vector>

// Scene point light objects (scene::kTypePointLight) → render::PointLight (world coordinates, both hierarchical and flattened scenes).
// Shared by the editor and the game view. Appends to out (brief lights from view-side effects are added after)
namespace mealkit::render {

inline Vec3 srgbToLinear(Color c) {
    auto ch = [](u32 v) {
        const f32 x = static_cast<f32>(v & 0xFFu) / 255.0f;
        return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
    };
    return {ch(c.value), ch(c.value >> 8), ch(c.value >> 16)};
}

inline void addSceneLights(const scene::Scene& scene, std::vector<PointLight>& out) {
    for (const scene::Entity& e : scene.entities) {
        if (!e.is(scene::kTypePointLight)) continue;
        const scene::PointLightFields f = scene::readObject<scene::PointLightFields>(e);
        if (f.intensity <= 0.0f || f.range <= 0.0f) continue;
        out.push_back({scene::worldTransform(scene, e).position, f.range, srgbToLinear(f.color) * f.intensity});
    }
}

} // namespace mealkit::render
