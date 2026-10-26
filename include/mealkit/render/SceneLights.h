#pragma once

#include "mealkit/render/Renderer.h"
#include "mealkit/scene/Scene.h"

#include <vector>

// 씬의 점광원 물체(scene::kTypePointLight) → render::PointLight (월드 좌표, 계층·펼친 씬 모두).
// 에디터와 게임 화면이 같이 쓴다. out에 더한다(화면 쪽 효과의 잠깐 빛은 그 뒤에 더한다)
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
