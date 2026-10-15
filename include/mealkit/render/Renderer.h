#pragma once

#include "mealkit/core/Math.h"
#include "mealkit/render/Mesh.h"

namespace mealkit::render {

using MeshId = u32;

// GPU로 그대로 복사되는 인스턴스 데이터 (HLSL Instance 구조체와 동일한 48바이트).
struct InstanceData {
    Vec3 position;
    u32 color = 0xFFFFFFFF; // rgba() 로 만든 sRGB 색
    Quat rotation;
    Vec3 scale{1, 1, 1};
    u32 flags = 0;
};
static_assert(sizeof(InstanceData) == 48, "InstanceData must match the HLSL layout");

inline constexpr u32 kInstanceChecker = 1u;   // 바닥용 체커 패턴
inline constexpr u32 kInstanceHighlight = 2u; // 에디터 선택 강조

constexpr u32 rgba(u8 r, u8 g, u8 b, u8 a = 255) {
    return static_cast<u32>(r) | (static_cast<u32>(g) << 8) | (static_cast<u32>(b) << 16) | (static_cast<u32>(a) << 24);
}

// 3D를 그릴 화면 영역(픽셀). width가 0이면 창 전체.
struct Viewport {
    f32 x = 0.0f, y = 0.0f, width = 0.0f, height = 0.0f;
};

struct FrameParams {
    Viewport viewport;
    Mat4 viewProj;
    Vec3 cameraPos;
    Vec3 sunDirection{0.4f, 0.8f, 0.3f}; // 태양을 향하는 방향 (정규화는 렌더러가 한다)
    Vec3 sunColor{1.0f, 0.95f, 0.85f};
    Vec3 skyColor{0.30f, 0.38f, 0.50f}; // 선형 색 공간
    Vec3 groundColor{0.20f, 0.17f, 0.14f};
    Vec3 fogColor{0.50f, 0.62f, 0.78f}; // 배경 지우기 색으로도 쓴다
    f32 fogDensity = 0.008f;
};

struct RendererDesc {
    void* window = nullptr; // HWND
    u32 width = 0;
    u32 height = 0;
    bool vsync = true;
    bool debugLayer = false;
};

struct RenderStats {
    u32 drawCalls = 0;
    u32 instances = 0;
};

// 에디터 UI 같은 확장 모듈용 원시 D3D12 핸들. commandList는 beginFrame~endFrame 사이에만 유효하다.
struct NativeHandles {
    void* device = nullptr;      // ID3D12Device*
    void* queue = nullptr;       // ID3D12CommandQueue*
    void* commandList = nullptr; // ID3D12GraphicsCommandList*
    u32 framesInFlight = 0;
    u32 overlayFormat = 0; // DXGI_FORMAT: beginOverlay()가 묶는 렌더 타깃 형식 (감마 변환 없음)
};

// DX12 포워드 렌더러. 프레임 2개를 동시에 처리하고, 프레임별 데이터는 계속 매핑된 업로드 링 버퍼에 쓴다.
// 내장 그래픽(UMA)에서는 업로드 힙이 곧 GPU가 읽는 메모리라 복사가 전혀 없다.
class Renderer {
public:
    Renderer() = default;
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool init(const RendererDesc& desc);
    void shutdown();
    void resize(u32 width, u32 height);

    MeshId createMesh(const MeshData& mesh);

    // false면 이번 프레임은 그리지 않는다 (최소화 등).
    bool beginFrame(const FrameParams& params);
    // 같은 메시의 인스턴스를 드로우 콜 1회로 그린다.
    void drawInstances(MeshId mesh, const InstanceData* instances, u32 count);
    // 3D 이후 UI를 그리기 위해 깊이 없는 sRGB 미변환 백버퍼 뷰를 묶는다.
    void beginOverlay();
    void endFrame();
    void waitIdle();

    NativeHandles native() const;
    u32 width() const;
    u32 height() const;

    void setVsync(bool enabled);
    bool vsync() const;
    void requestCapture(const char* pngPath); // 다음 endFrame에서 백버퍼를 PNG로 저장

    const char* adapterName() const;
    bool isUma() const;
    const RenderStats& stats() const;

private:
    struct Impl;
    Impl* m = nullptr;
};

} // namespace mealkit::render
