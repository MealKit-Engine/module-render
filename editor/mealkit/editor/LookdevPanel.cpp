#include "mealkit/editor/LookdevPanel.h"
#include "mealkit/editor/FieldEditor.h"

#include "mealkit/editor/Strings.h"
#include "mealkit/render/Environment.h"

#include <algorithm>
#include <cmath>

#include <imgui.h>

namespace mealkit::editor {
namespace {

constexpr u32 kTargetSize = 768; // square image (scaled to the window size)
constexpr f32 kFovY = 35.0f * 3.14159265f / 180.0f;
constexpr f32 kSpinSpeed = 0.5f; // stand rotation speed (rad/s)

// Color lighting presets: sets of sun/sky/ground/background colors. After these in the list come sky HDRIs from .hdr files in assets/textures (sun from the neutral studio one)
struct LightPreset {
    Vec3 sunDirection, sunColor, sky, ground, background;
};
const LightPreset kPresets[] = {
    {{0.45f, 0.75f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.32f, 0.32f, 0.34f}, {0.16f, 0.16f, 0.16f},
     {0.20f, 0.20f, 0.22f}},
    {{0.4f, 0.8f, 0.3f}, {1.0f, 0.95f, 0.85f}, {0.18f, 0.36f, 0.85f}, {0.20f, 0.17f, 0.14f},
     {0.55f, 0.70f, 0.95f}},
    {{-0.7f, 0.25f, -0.4f}, {1.0f, 0.55f, 0.30f}, {0.22f, 0.18f, 0.32f}, {0.10f, 0.07f, 0.05f},
     {0.32f, 0.22f, 0.28f}},
    {{0.2f, 0.6f, -0.6f}, {0.35f, 0.45f, 0.75f}, {0.06f, 0.08f, 0.14f}, {0.02f, 0.02f, 0.03f},
     {0.03f, 0.04f, 0.07f}},
};
constexpr u32 kPresetCount = static_cast<u32>(std::size(kPresets));

// Preset names (the editor text check looks for tr() source strings, so each is called once here)
const char* presetName(u32 i) {
    const char* names[kPresetCount] = {tr("Neutral studio"), tr("Daylight"), tr("Sunset"), tr("Night")};
    return names[std::min(i, kPresetCount - 1)];
}

} // namespace

void LookdevPanel::render(render::Renderer& renderer, asset::ModelCache& models, render::LightingModel projectLighting, f64 dt) {
    if (!m_target) m_target = renderer.createRenderTarget(kTargetSize, kTargetSize);
    if (!m_target) return;
    if (m_spin) m_yaw += static_cast<f32>(dt) * kSpinSpeed;

    const asset::Model* model = m_model.empty() ? nullptr : models.get(m_model);
    m_hasModel = model != nullptr;
    if (!m_hdrisListed) {
        m_hdris = render::listHdris(asset::ModelCache::assetsRoot());
        m_hdrisListed = true;
    }
    const bool hdri = m_preset >= kPresetCount && m_preset - kPresetCount < m_hdris.size();
    const LightPreset& light = kPresets[hdri ? 0 : std::min(m_preset, kPresetCount - 1)];

    // Camera distance is set to fit the model bounding box (centered on the stand)
    Vec3 lo{-0.5f, 0.0f, -0.5f}, hi{0.5f, 1.0f, 0.5f};
    if (model) lo = model->boundsMin, hi = model->boundsMax;
    const Vec3 size = hi - lo;
    const f32 radius = std::max(0.5f * std::sqrt(size.x * size.x + size.y * size.y + size.z * size.z), 0.05f);
    const Vec3 center{0.0f, 0.5f * size.y, 0.0f};
    const f32 distance = radius / std::sin(kFovY * 0.5f) * 1.3f / m_zoom;
    const Vec3 eye = center + Vec3{0.0f, std::sin(m_pitch), -std::cos(m_pitch)} * distance;

    render::FrameParams frame;
    frame.viewProj = lookAtLH(eye, center, {0, 1, 0}) * perspectiveReversedZ(kFovY, 1.0f, distance * 0.02f, distance * 4.0f);
    frame.cameraPos = eye;
    frame.cameraForward = normalize(center - eye);
    frame.sunDirection = light.sunDirection;
    frame.sunColor = light.sunColor;
    frame.skyColor = light.sky;
    frame.groundColor = light.ground;
    frame.fogColor = light.background;
    frame.fogDensity = 0.0f;
    if (hdri) frame.environment = models.environment(m_hdris[m_preset - kPresetCount]);
    frame.time = static_cast<f32>(std::fmod(ImGui::GetTime(), 4096.0)); // wind sway

    m_drawList.clear();
    const Quat spin = quatFromAxisAngle({0, 1, 0}, m_yaw);
    // Stand: a flat box under the model's feet (rotates with it)
    const f32 plinth = std::max(size.x, size.z) * 0.6f + radius * 0.15f; // radius
    m_drawList.add(models.boxMesh(), {Vec3{0.0f, -radius * 0.05f, 0.0f}, render::rgba(150, 154, 166), spin,
                                      Vec3{plinth * 2.0f, radius * 0.1f, plinth * 2.0f}, 0});
    if (model && model->lodCount) {
        const Vec3 base{-(lo.x + hi.x) * 0.5f, -lo.y, -(lo.z + hi.z) * 0.5f}; // bounding box bottom center onto the stand
        const u32 flags = render::instanceDissolve(render::instanceFlash(0, m_flash), m_dissolve);
        m_drawList.add(model->lods[0], {rotate(spin, base), 0xFFFFFFFF, spin, Vec3{1, 1, 1}, flags});
    }
    const render::LightingModel lighting =
        m_lighting < 0 ? projectLighting : static_cast<render::LightingModel>(std::min<i32>(m_lighting, render::kLightingModelCount - 1));
    renderer.drawToTarget(m_target, frame, m_drawList, lighting);
}

void LookdevPanel::drawContent(render::Renderer& renderer) {
    const bool hdri = m_preset >= kPresetCount && m_preset - kPresetCount < m_hdris.size();
    if (beginCombo(tr("Light"), hdri ? m_hdris[m_preset - kPresetCount].c_str() : presetName(m_preset))) {
        if (ImGui::IsWindowAppearing()) m_hdris = render::listHdris(asset::ModelCache::assetsRoot()); // newly added HDRIs
        for (u32 i = 0; i < kPresetCount; ++i)
            if (ImGui::Selectable(presetName(i), m_preset == i)) m_preset = i;
        if (!m_hdris.empty()) ImGui::Separator();
        for (u32 i = 0; i < m_hdris.size(); ++i)
            if (ImGui::Selectable(m_hdris[i].c_str(), m_preset == kPresetCount + i)) m_preset = kPresetCount + i;
        ImGui::EndCombo();
    }
    const char* lightingNames[] = {tr("Project setting"), "Pbr", "Toon", "Unlit"};
    i32 lighting = m_lighting + 1;
    if (ImGui::Combo(tr("Lighting model"), &lighting, lightingNames, static_cast<int>(std::size(lightingNames)))) m_lighting = lighting - 1;
    ImGui::Checkbox(tr("Turn"), &m_spin);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderFloat(tr("Zoom"), &m_zoom, 0.5f, 3.0f, "%.2f");
    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderFloat(tr("Flash"), &m_flash, 0.0f, 1.0f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderFloat(tr("Dissolve"), &m_dissolve, 0.0f, 1.0f, "%.2f");

    if (m_model.empty()) ImGui::TextDisabled("%s", tr("Pick a model in [Assets]."));
    else if (!m_hasModel) ImGui::TextDisabled("%s", trf("Cannot load %s", m_model.c_str()).c_str());
    else ImGui::TextDisabled("%s", m_model.c_str());

    // Image: a square fitting the remaining space. Dragging rotates the stand and changes elevation
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const f32 side = std::max(64.0f, std::min(avail.x, avail.y));
    const rhi::Texture texture = renderer.renderTargetTexture(m_target);
    if (texture) {
        ImGui::Image(ImTextureRef(static_cast<ImTextureID>(rhi::imguiTextureId(renderer.device(), texture))), ImVec2(side, side));
        if (ImGui::IsItemHovered()) {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                const ImVec2 d = ImGui::GetIO().MouseDelta;
                m_yaw -= d.x * 0.01f;
                m_pitch = std::clamp(m_pitch + d.y * 0.01f, -0.3f, 1.4f);
                m_spin = false;
            }
            const f32 wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.0f) m_zoom = std::clamp(m_zoom * (wheel > 0.0f ? 1.1f : 1.0f / 1.1f), 0.5f, 3.0f);
        }
    }
}

} // namespace mealkit::editor
