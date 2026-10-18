#pragma once

#include "mealkit/core/Math.h"
#include "mealkit/render/Mesh.h"

#include <string>

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
    Vec3 cameraForward{0.0f, 0.0f, 1.0f}; // 소리 리스너 방향 (엔진이 프레임마다 audio::Engine::setListener에 넘긴다)
    Vec3 sunDirection{0.4f, 0.8f, 0.3f}; // 태양을 향하는 방향 (정규화는 렌더러가 한다)
    Vec3 sunColor{1.0f, 0.95f, 0.85f};
    Vec3 skyColor{0.30f, 0.38f, 0.50f}; // 선형 색 공간
    Vec3 groundColor{0.20f, 0.17f, 0.14f};
    Vec3 fogColor{0.50f, 0.62f, 0.78f}; // 배경 지우기 색으로도 쓴다
    f32 fogDensity = 0.008f;
    // 화면 전체를 이 색(선형)으로 섞는다 (연출 표의 번쩍임, 엔진이 feedback::Player에서 채운다). 0 = 없음
    Vec3 tintColor{1.0f, 1.0f, 1.0f};
    f32 tintAmount = 0.0f;
};

struct RendererDesc {
    void* window = nullptr; // HWND
    u32 width = 0;
    u32 height = 0;
    bool vsync = true;
    bool debugLayer = false;
};

// GPU 기능 조사 결과. 시작 로그와 `--gpu-info`(최소 사양 실기 측정 기록)에 쓴다.
struct GpuInfo {
    std::string adapter;       // 어댑터 이름 (UTF-8)
    std::string driverVersion; // 사용자 모드 드라이버 버전 "31.0.101.5186" (모르면 빈 문자열)
    u32 vendorId = 0;          // 0x8086 Intel, 0x1002 AMD, 0x10DE NVIDIA
    u32 deviceId = 0;
    u64 dedicatedVideoMemory = 0;
    u64 sharedSystemMemory = 0;
    u32 featureLevel = 0;        // D3D_FEATURE_LEVEL (0xC100 = 12_1)
    u32 shaderModel = 0;         // D3D_SHADER_MODEL (0x66 = 6.6)
    u32 resourceBindingTier = 0; // 1~3
    u32 meshShaderTier = 0;      // 0 = 없음 (참고용, 엔진은 쓰지 않는다)
    u32 waveLaneCountMin = 0;
    u32 waveLaneCountMax = 0;
    bool uma = false; // 내장 그래픽처럼 CPU와 메모리를 같이 쓴다
};

// 최소 사양에 못 미치는 이유. 맞으면 빈 문자열.
std::string gpuMinimumError(const GpuInfo& info);

// 창 없이 렌더러가 고를 GPU를 조사한다. GPU가 없으면 false와 error.
bool queryGpu(GpuInfo& out, std::string* error = nullptr);

struct RenderStats {
    u32 drawCalls = 0;
    u32 instances = 0;
};

// 2D UI 사각형 하나 (HLSL UiQuad와 같은 112바이트, shaders/Ui.hlsl). mealkit/ui/가 채운다. 좌표는 백버퍼 픽셀.
struct UiQuad {
    f32 rect[4] = {};                    // x, y, w, h (회전 전 상자)
    f32 clip[4] = {-1e9f, -1e9f, 1e9f, 1e9f}; // x0, y0, x1, y1 (이 밖은 그리지 않는다)
    f32 radii[4] = {};                   // 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래
    f32 uv[4] = {};                      // 아틀라스 u0, v0, u1, v1 (이미지·글자)
    f32 gradient[4] = {};                // 선형: 시작 xy, 끝 xy / 원형: 중심 xy, 반지름 xy (상자 0..1 좌표)
    u32 color0 = 0;                      // sRGB RGBA8 (rgba()), 곧은 알파
    u32 color1 = 0;
    u32 strokeColor = 0;
    u32 flags = 0; // kUiShape/Image/Glyph/Shadow | kUiLinear/kUiRadial
    f32 strokeWidth = 0.0f;
    f32 rotation = 0.0f; // 라디안, 상자 가운데 기준
    f32 softness = 0.0f; // 그림자 흐림 (픽셀)
    f32 sdfScale = 0.0f; // 글자: SDF 값 1당 화면 픽셀
};
static_assert(sizeof(UiQuad) == 112, "UiQuad must match the HLSL layout");

inline constexpr u32 kUiShape = 0, kUiImage = 1, kUiGlyph = 2, kUiShadow = 3;
inline constexpr u32 kUiLinear = 1u << 4, kUiRadial = 2u << 4;

enum class UiAtlas : u8 { Glyphs, Images }; // 글자 SDF (1바이트), UI 이미지 (RGBA 4바이트)
inline constexpr u32 kUiAtlasSize = 2048;

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
    // UI 아틀라스의 한 영역을 바꾼다 (언제든, 다음 drawUi 전에 GPU에 올린다). pixels는 w×h, 행 사이 빈틈 없음
    void updateUiAtlas(UiAtlas atlas, u32 x, u32 y, u32 w, u32 h, const u8* pixels);
    // 2D UI를 그린다: 3D 다음, 백버퍼 원래 해상도(동적 해상도와 무관), 깊이 없음. 드로우 콜 1회
    void drawUi(const UiQuad* quads, u32 count);
    void endFrame();
    void waitIdle();

    NativeHandles native() const;
    u32 width() const;
    u32 height() const;

    void setVsync(bool enabled);
    bool vsync() const;
    void requestCapture(const char* pngPath); // 다음 endFrame에서 백버퍼를 PNG로 저장

    const char* adapterName() const;
    const GpuInfo& gpu() const;
    const char* initError() const; // init()이 false일 때 이유
    bool isUma() const;
    const RenderStats& stats() const;

private:
    struct Impl;
    Impl* m = nullptr;
};

} // namespace mealkit::render
