#pragma once

#include "mealkit/asset/ModelCache.h"
#include "mealkit/render/DrawList.h"
#include "mealkit/render/Renderer.h"

#include <string>
#include <vector>

// [룩데브] 창 (docs/ROADMAP.md "텍스처와 머티리얼 3부"): 씬 조명과 상관없이 고정 조명 + 회전 받침대에서 모델·머티리얼을 본다.
// 오프스크린 대상(rhi::RenderTarget)에 그려 창 안에 그림으로 띄운다. 조명 모델과 조명(색 프리셋 넷 + assets/textures의 .hdr의 하늘 HDRI),
// 머티리얼 기능 미리보기(번쩍임·디졸브 진행도)를 바꿔 본다. 게임에는 들어가지 않는다 (에디터 빌드만).
namespace mealkit::editor {

class LookdevPanel {
public:
    static constexpr const char* kWindow = "Look Dev###LookDev";
    // 보여 줄 모델 (씬에서 고른 물체의 모델, 비면 안내만)
    void setModel(const std::string& path) { m_model = path; }
    const std::string& model() const { return m_model; }
    // 3D를 그린 뒤, UI 전에 (창이 열려 있을 때만). 대상은 처음 부를 때 만든다
    void render(render::Renderer& renderer, asset::ModelCache& models, render::LightingModel projectLighting, f64 dt);
    // ImGui 창 (Editor::renderUi 안에서)
    void draw(render::Renderer& renderer, bool* open);
    // 자동 캡처용 (--lookdev): 조명 번호 (색 프리셋 다음이 HDRI 목록), 조명 모델 번호 (-1 = 프로젝트 설정)
    void setPreset(u32 preset) { m_preset = preset; }
    void setLighting(i32 lighting) { m_lighting = lighting; }

private:
    rhi::RenderTarget m_target;
    render::DrawList m_drawList;
    std::string m_model;
    u32 m_preset = 0;            // 색 프리셋 번호, 그 뒤는 m_hdris 번호 (프리셋 수만큼 뺀다)
    std::vector<std::string> m_hdris; // assets/textures의 .hdr (조명 목록을 열 때 다시 찾는다)
    bool m_hdrisListed = false;
    i32 m_lighting = -1; // -1 = 프로젝트 설정, 0.. = render::LightingModel
    f32 m_yaw = 0.6f;    // 받침대 회전 (라디안)
    f32 m_pitch = 0.25f; // 카메라 높이각
    f32 m_zoom = 1.0f;
    bool m_spin = true;
    f32 m_flash = 0.0f;
    f32 m_dissolve = 0.0f;
    bool m_hasModel = false; // 지난 render에서 모델을 찾았다
};

} // namespace mealkit::editor
