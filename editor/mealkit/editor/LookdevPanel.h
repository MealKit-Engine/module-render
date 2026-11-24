#pragma once

#include "mealkit/asset/ModelCache.h"
#include "mealkit/render/DrawList.h"
#include "mealkit/render/Renderer.h"

#include <string>
#include <vector>

// [Lookdev] window (docs/ROADMAP.md "Textures and materials part 3"): view models/materials under fixed lighting on a rotating stand, independent of scene lighting.
// Draws to an offscreen target (rhi::RenderTarget) and shows it as an image in the window. Switch the lighting model and lighting (four color presets + sky HDRIs from .hdr in assets/textures),
// and preview material features (flash/dissolve progress). Not included in the game (editor builds only).
namespace mealkit::editor {

class LookdevPanel {
public:
    // Model to show (model of the object selected in the scene; only a hint if empty)
    void setModel(const std::string& path) { m_model = path; }
    const std::string& model() const { return m_model; }
    // After drawing 3D, before UI (only when the window is open). The target is created on the first call
    void render(render::Renderer& renderer, asset::ModelCache& models, render::LightingModel projectLighting, f64 dt);
    // ImGui window (inside Editor::renderUi)
    void drawContent(render::Renderer& renderer);
    // [Materials] page (T4): read-only list of the shown model's materials; the selected row's triangles are tinted in the image
    void drawMaterials();
    void drawView();
    void drawImage(render::Renderer& renderer);
    // For automated capture (--lookdev): lighting index (HDRI list after the color presets), lighting model index (-1 = project setting)
    void setPreset(u32 preset) { m_preset = preset; }
    void setLighting(i32 lighting) { m_lighting = lighting; }
    // --lookdev-material <n>: open the [Materials] page with the n-th material of the model selected (its triangles tinted)
    void selectMaterial(i32 index) { m_material = index; m_openMaterials = true; }

private:
    rhi::RenderTarget m_target;
    render::DrawList m_drawList;
    std::string m_model;
    u32 m_preset = 0;            // color preset index, beyond that an m_hdris index (minus the preset count)
    std::vector<std::string> m_hdris; // .hdr in assets/textures (rescanned when the lighting list opens)
    bool m_hdrisListed = false;
    i32 m_lighting = -1; // -1 = project setting, 0.. = render::LightingModel
    f32 m_yaw = 0.6f;    // stand rotation (radians)
    f32 m_pitch = 0.25f; // camera elevation
    f32 m_zoom = 1.0f;
    bool m_spin = true;
    f32 m_flash = 0.0f;
    f32 m_dissolve = 0.0f;
    bool m_hasModel = false; // the last render found a model
    const std::vector<asset::MaterialInfo>* m_info = nullptr; // the shown model's materials (set by render(), valid for this frame's UI)
    i32 m_material = -1;     // selected row on the [Materials] page (-1 = none): tinted in the image
    bool m_materialsShown = false; // the [Materials] page was drawn last frame
    bool m_openMaterials = false; // select the [Materials] tab on the next draw (--lookdev-material)
};

} // namespace mealkit::editor
