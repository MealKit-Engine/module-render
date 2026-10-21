#pragma once

#include "mealkit/core/Math.h"
#include "mealkit/render/Culling.h"
#include "mealkit/render/Mesh.h"
#include "mealkit/rhi/Rhi.h"

#include <string>

namespace mealkit::render {

using MeshId = u32;

// 머티리얼 (glTF PBR metallic-roughness의 지원 범위, docs/CODING_MODEL.md "모델·머티리얼·텍스처"). 렌더러가 표로 갖고
// 프레임마다 업로드 링에 쓴다 (shaders/Forward.hlsl Material, 64바이트). 텍스처는 bindless 번호 (kNoTexture = 없음)
inline constexpr u32 kNoTexture = 0xFFFFFFFFu;
struct MaterialData {
    f32 baseColor[4] = {1, 1, 1, 1}; // 선형
    f32 emissive[3] = {0, 0, 0};
    f32 metallic = 0.0f;
    f32 roughness = 1.0f;
    f32 normalScale = 1.0f;
    f32 alphaCutoff = 0.0f; // > 0이면 알파 마스크 (이보다 작은 알파는 그리지 않는다)
    u32 baseColorTexture = kNoTexture;
    u32 normalTexture = kNoTexture;
    u32 metallicRoughnessTexture = kNoTexture; // G = 거칠기, B = 금속 (glTF)
    u32 emissiveTexture = kNoTexture;
};

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
inline constexpr u32 kInstanceOverBudget = 4u; // 에디터 보기 모드 Budget: 에셋 예산을 넘은 모델
inline constexpr u32 kInstanceLodShift = 8;    // 에디터 보기 모드 Lod: 비트 8~9 = 그린 LOD 단계

// 에디터 보기 모드 (씬 뷰에서 무엇이 비싼지 색으로). 디버그 UI가 있는 빌드에만 파이프라인이 있고, 출시 빌드는 None만
enum class DebugView : u8 {
    None,
    Wireframe, // 삼각형 선
    Normals,   // 법선 방향 = 색
    Lod,       // LOD 0 초록, 1 노랑, 2 빨강
    Overdraw,  // 같은 픽셀을 그린 횟수 = 밝기 (깊이 시험 없이 더하기)
    Budget,    // 예산을 넘은 모델 빨강, 나머지 회색
    Count
};

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
    DebugView debugView = DebugView::None; // 에디터 보기 모드
};

struct RendererDesc {
    void* window = nullptr; // HWND
    u32 width = 0;
    u32 height = 0;
    bool vsync = true;
    bool debugLayer = false;
    bool gpuCapture = false; // --gpu-capture (rhi/GpuCapture.h)
};

// GPU 조사는 RHI 백엔드가 한다 (출시 exe의 --system-info, 최소 사양 검사)
using GpuInfo = rhi::GpuInfo;
using rhi::gpuMinimumError;
using rhi::queryGpu;

// GPU 패스: 프레임 GPU 시간을 나눠 재는 구간 (벤치마크 보고서의 GPU 예산 표, mealkit.project gpuBudget*).
// 아직 없는 패스(그림자·포스트·VFX)는 그 기능이 들어올 때 렌더러가 재기 시작한다 (그때까지 0)
enum class GpuPass : u8 { Opaque, Shadow, Post, Vfx, Ui, Count };
inline constexpr u32 kGpuPassCount = static_cast<u32>(GpuPass::Count);
const char* gpuPassName(GpuPass pass); // "opaque", "shadow", ... (보고서·로그, 영어)

struct RenderStats {
    u32 drawCalls = 0;
    u32 instances = 0;
    u32 culled = 0; // 프러스텀 밖이라 빠진 인스턴스 (DrawList::submit)
    // GPU 시간 (타임스탬프): GPU가 끝낸 가장 최근 프레임 값이라 2프레임 늦다. gpuValid가 false면 모른다
    bool gpuValid = false;
    f32 gpuFrameMs = 0.0f;
    f32 gpuPassMs[kGpuPassCount] = {};
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

// 포워드 렌더러 (RHI 위, 그래픽 API를 직접 부르지 않는다). 프레임 2개를 동시에 처리하고, 프레임별 데이터는
// RHI 업로드 링에 쓴다. 내장 그래픽(UMA)에서는 업로드 메모리가 곧 GPU가 읽는 메모리라 복사가 전혀 없다.
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
    // 정점의 material 번호가 가리킨다. 0번은 기본 흰색 (init이 만든다)
    u32 createMaterial(const MaterialData& material);
    // 밉이 있는 텍스처 (asset::TextureCache). 셰이더가 읽는 bindless 번호
    u32 createTexture(const rhi::TextureInit& init);
    // 메시 로컬 바운딩 구 (createMesh가 정점에서 구한다). 없는 메시는 반지름 0
    BoundingSphere meshBounds(MeshId mesh) const;
    // 이번 프레임 viewProj의 프러스텀 (beginFrame이 만든다). DrawList::submit이 인스턴스를 거른다
    const Frustum& frustum() const;
    void addCulled(u32 count); // 통계용 (DrawList::submit)

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

    rhi::Device& device(); // 디버그 UI 그리기 (rhi::imgui*)
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
    bool supportsDebugView(DebugView view) const; // 이 빌드·장치가 그 보기 모드를 그릴 수 있다
    rhi::GpuMemory gpuMemory() const; // 드라이버 질의라 매 프레임 부르지 않는다 (FrameStats는 보고 주기마다)

private:
    struct Impl;
    Impl* m = nullptr;
};

} // namespace mealkit::render
