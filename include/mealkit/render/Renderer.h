#pragma once

#include "mealkit/core/Math.h"
#include "mealkit/render/Culling.h"
#include "mealkit/render/Mesh.h"
#include "mealkit/rhi/Rhi.h"

#include <cmath>
#include <string>

namespace mealkit {
struct Image;
}

namespace mealkit::render {

class DrawList;
using MeshId = u32;

// 머티리얼 (glTF PBR metallic-roughness의 지원 범위, docs/CODING_MODEL.md "모델·머티리얼·텍스처"). 렌더러가 표로 갖고
// 프레임마다 업로드 링에 쓴다 (shaders/Forward.hlsl Material, 64바이트). 텍스처는 bindless 번호 (kNoTexture = 없음)
inline constexpr u32 kNoTexture = 0xFFFFFFFFu;

// 조명 모델 (스타일 프리셋, docs/CODING_MODEL.md "셰이더 코드"): 프로젝트 설정 lightingModel 하나가 게임 전체를 정한다.
// 셰이더는 모델마다 하나씩 빌드 때 컴파일되고, 렌더러가 init에서 파이프라인을 모두 만든다 (바꿔도 끊기지 않는다)
enum class LightingModel : u8 { Pbr, Toon, Unlit }; // 프로젝트 설정 game::LightingModelSetting과 같은 순서
inline constexpr u32 kLightingModelCount = 3;

// 머티리얼 기능 (glTF extras "mealkit_<키>", asset::modelSchemaJson의 materialFeatures). 셰이더 변형이 아니라
// 머티리얼 표의 비트로 셰이더 안에서 고른다: 한 메시에 기능이 다른 머티리얼이 섞여도 드로우 콜 1회 그대로
inline constexpr u32 kMaterialOutline = 1u;  // 외곽선: 뒤집은 껍질 패스 (메시에 하나라도 있으면 그 메시를 한 번 더 그린다)
inline constexpr u32 kMaterialWind = 2u;     // 바람 흔들림: 세기는 정점 색 알파로 칠한다 (알파 없는 모델은 1)
inline constexpr u32 kMaterialRim = 4u;      // 림 라이트: 가장자리가 밝아진다
inline constexpr u32 kMaterialDissolve = 8u; // 디졸브: 인스턴스의 디졸브 진행도(instanceDissolve)만큼 사라진다

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
    // 스타일 프리셋 머티리얼 기능 (kMaterial* 비트와 값). 색은 선형
    u32 features = 0;
    f32 outlineWidth = 0.02f; // 미터 (월드 단위, 인스턴스 크기와 무관)
    f32 outlineColor[3] = {0, 0, 0};
    f32 windStrength = 0.1f;  // 미터 (정점 알파 1, FrameParams::windStrength 1일 때 흔들리는 폭)
    f32 rimStrength = 0.5f;
    f32 rimColor[3] = {1, 1, 1};
    f32 dissolveEdge = 0.05f; // 사라지는 경계의 빛나는 폭 (진행도 단위 0..1)
    f32 dissolveColor[3] = {4.0f, 1.6f, 0.4f}; // 경계 발광 (선형, 1보다 크면 더 밝다)
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
// 인스턴스마다 바뀌는 스타일 값 (flags 위 16비트, 인스턴스는 48바이트 그대로): view/가 틱 이벤트로 바꾼다
inline constexpr u32 kInstanceFlashShift = 16;    // 비트 16~23: 번쩍임 0..1 (흰색으로 섞는다, 모든 머티리얼)
inline constexpr u32 kInstanceDissolveShift = 24; // 비트 24~31: 디졸브 진행도 0..1 (kMaterialDissolve 머티리얼만)
inline u32 instanceUnit8(f32 amount) { return static_cast<u32>((amount < 0.0f ? 0.0f : amount > 1.0f ? 1.0f : amount) * 255.0f + 0.5f); }
// flags에 번쩍임을 넣는다: out.add(mesh, {..., instanceFlash(flags, 0.8f)})
inline u32 instanceFlash(u32 flags, f32 amount) { return (flags & ~(0xFFu << kInstanceFlashShift)) | (instanceUnit8(amount) << kInstanceFlashShift); }
inline u32 instanceDissolve(u32 flags, f32 progress) {
    return (flags & ~(0xFFu << kInstanceDissolveShift)) | (instanceUnit8(progress) << kInstanceDissolveShift);
}

// 에디터 보기 모드 (씬 뷰에서 무엇이 비싼지 색으로). 디버그 UI가 있는 빌드에만 파이프라인이 있고, 출시 빌드는 None만
enum class DebugView : u8 {
    None,
    Wireframe, // 삼각형 선
    Normals,   // 법선 방향 = 색
    Lod,       // LOD 0 초록, 1 노랑, 2 빨강
    Overdraw,  // 같은 픽셀을 그린 횟수 = 밝기 (깊이 시험 없이 더하기)
    Budget,    // 예산을 넘은 모델 빨강, 나머지 회색
    Features,  // 머티리얼 기능: 외곽선 빨강, 바람 초록, 림 파랑, 디졸브 노랑 (섞이면 더한 색), 없음 회색
    Count
};

constexpr u32 rgba(u8 r, u8 g, u8 b, u8 a = 255) {
    return static_cast<u32>(r) | (static_cast<u32>(g) << 8) | (static_cast<u32>(b) << 16) | (static_cast<u32>(a) << 24);
}

// 3D를 그릴 화면 영역(픽셀). width가 0이면 창 전체.
struct Viewport {
    f32 x = 0.0f, y = 0.0f, width = 0.0f, height = 0.0f;
};

// 반사 프로브 하나 (씬의 reflection_probe 물체, asset::ModelCache::probes가 채운다). 물체는 자기 위치를 담은 영향 상자
// 중 중심이 가장 가까운 프로브 하나의 반사 큐브를 쓴다 (섞지 않음). 어느 상자에도 없으면 하늘 HDRI 반사
struct ReflectionProbe {
    Vec3 position;     // 구운 자리 = 영향 상자 중심
    Vec3 halfExtents;  // 영향 상자 (축 정렬, 월드)
    u32 id = 0; // Renderer::createProbe 번호 (0 = 없음, 건너뛴다)
};
inline constexpr u32 kMaxReflectionProbes = 64; // 셰이더가 물체마다 모두 훑는다 (씬당 한도는 프로젝트 설정 probeMaxCount)

// 점광원 하나 (씬의 point_light 물체 = render::addSceneLights, 화면 쪽 효과의 잠깐 빛은 view/가 더한다). 그림자는 없다.
// 렌더러가 물체(인스턴스)마다 영향이 큰 빛을 CPU로 최대 setLightsPerObject개 골라 셰이더에 번호를 넘긴다
struct PointLight {
    Vec3 position;
    f32 range = 6.0f;   // 이 거리에서 0이 된다 (미터)
    Vec3 color{1, 1, 1}; // 선형 색 × 세기
};
inline constexpr u32 kMaxPointLights = 255;      // 프레임당 (번호 8비트, 넘으면 앞에서부터)
inline constexpr u32 kMaxLightsPerObject = 8;    // 인스턴스마다 번호 8개 (8바이트)

// 태양 그림자 (CSM, docs/ROADMAP.md 2단계 "그림자"): 카메라 앞 distance까지를 단계 1~2개로 나눠 단계마다 size² 그림자 맵.
// 프로젝트 설정 shadows*가 정한다(game::applyRenderStyle). 점광원은 그림자가 없다
struct ShadowSettings {
    bool enabled = true;
    u32 cascades = 2;     // 1 또는 2
    u32 size = 1024;      // 단계 하나의 그림자 맵 크기 (512..4096)
    f32 distance = 40.0f; // 카메라에서 이 거리(미터)까지 그림자 (끝 10%에서 옅어진다)
};
inline constexpr u32 kMaxShadowCascades = 2;

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
    // 화면 시간 (초, 바람 흔들림). 그래픽 회귀 테스트는 0으로 둔다
    f32 time = 0.0f;
    Vec3 windDirection{1.0f, 0.0f, 0.3f}; // 수평 방향 (정규화는 렌더러가 한다)
    f32 windStrength = 1.0f;              // 머티리얼 windStrength에 곱한다 (0 = 바람 없음)
    // 하늘 HDRI (Renderer::createEnvironment 번호, 0 = 없음: 하늘·땅 색 그라디언트와 안개 색 배경). 있으면 스카이박스를 그리고
    // 환경광(SH)·반사(큐브 밉 = 거칠기)를 그 HDRI에서 읽는다. 세기는 하늘·환경광·반사 모두에 곱하고, 회전은 y축 (라디안)
    u32 environment = 0;
    f32 environmentIntensity = 1.0f;
    f32 environmentRotation = 0.0f;
    // 반사 프로브 (부르는 쪽 메모리, beginFrame/drawToTarget 동안만 읽는다). kMaxReflectionProbes를 넘으면 앞에서부터
    const ReflectionProbe* probes = nullptr;
    u32 probeCount = 0;
    // 점광원 (부르는 쪽 메모리, 프레임 동안 읽는다). kMaxPointLights를 넘으면 앞에서부터
    const PointLight* lights = nullptr;
    u32 lightCount = 0;
};

// 하늘 HDRI에서 구운 환경 (asset::EnvironmentMap, 큐브는 RGB9E5, 면 +X −X +Y −Y +Z −Z 차례로). createEnvironment가 GPU에 올린다
struct EnvironmentData {
    f32 sh[9][3] = {}; // 환경광 색 = Σ sh[i] · Y_i(n)
    u32 skySize = 0;
    const u32* sky = nullptr; // 면마다 skySize²
    u32 reflectSize = 0, reflectMips = 0;
    const u32* reflect = nullptr; // 면마다 밉 0부터
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

    // 하늘 HDRI 환경을 GPU에 올리고 번호를 준다 (FrameParams::environment). 지우지 않는다: 같은 HDRI는 쓰는 쪽(ModelCache)이 번호를 다시 쓴다
    u32 createEnvironment(const EnvironmentData& data);
    // 반사 프로브 큐브 (asset::ProbeMap: RGB9E5, 면마다 밉 0부터)를 올리고 번호를 준다 (ReflectionProbe::id). 지우지 않는다
    u32 createProbe(u32 size, u32 mips, const u32* reflect);
    // 조명 모델 (프로젝트 설정, game::applyRenderStyle). 파이프라인은 init이 모두 만들어 두어 바로 바뀐다
    void setLightingModel(LightingModel model);
    LightingModel lightingModel() const;
    // 툰 램프: 가로 그림의 왼쪽(어두운 면) → 오른쪽(밝은 면) 색을 확산 조명에 곱한다 (첫 줄만 쓴다). nullptr = 엔진 기본 3단계
    void setToonRamp(const Image* ramp);
    // 물체마다 쓰는 점광원 수 (프로젝트 설정 pointLightsPerObject, 1..kMaxLightsPerObject, 기본 4)
    void setLightsPerObject(u32 count);
    // 태양 그림자 (프로젝트 설정). 크기·단계 수가 바뀌면 그림자 맵을 새로 만든다 (옛 것은 남는다)
    void setShadows(const ShadowSettings& shadows);
    const ShadowSettings& shadows() const;
    // 그림자 패스: DrawList::submit이 본 화면 3D보다 먼저 부른다 (이 프레임에 한 번, 그림자가 꺼졌으면 아무것도 하지 않는다).
    // 프러스텀 컬링 전의 모든 인스턴스가 그림자를 드리운다 (단계마다 빛 쪽 상자로 거른다)
    void drawShadows(const DrawList& list);

    // false면 이번 프레임은 그리지 않는다 (최소화 등).
    bool beginFrame(const FrameParams& params);
    // 같은 메시의 인스턴스를 드로우 콜 1회로 그린다.
    void drawInstances(MeshId mesh, const InstanceData* instances, u32 count);
    // 스킨 메시 하나 (캐릭터마다 드로우 콜 1회): bones는 asset::skinPalette의 팔레트 (관절 수, 최대 64)
    void drawSkinned(MeshId mesh, const InstanceData& instance, const BoneMatrix* bones, u32 boneCount);
    bool meshSkinned(MeshId mesh) const;
    // 오프스크린 그림 (에디터 룩데브 창): 크기를 정해 한 번 만들고 계속 쓴다 (지우지 않는다)
    rhi::RenderTarget createRenderTarget(u32 width, u32 height);
    rhi::Texture renderTargetTexture(rhi::RenderTarget target) const; // ImGui::Image (rhi::imguiTextureId)
    // 본 화면 3D(DrawList::submit) 다음, UI 전에: 대상 전체에 params의 카메라·빛과 lighting 조명 모델로 list를 그린다.
    // 백버퍼는 풀리므로 이 뒤에 본 화면 3D를 그리지 않는다 (UI는 beginOverlay가 다시 묶는다)
    void drawToTarget(rhi::RenderTarget target, const FrameParams& params, const DrawList& list, LightingModel lighting);
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
    // 이번 프레임 endFrame에서 오프스크린 대상(drawToTarget으로 그린)을 읽어 out에 둔다 (sRGB 8비트, GPU를 기다린다).
    // 반사 프로브 굽기. out은 endFrame까지 살아 있어야 한다. 같은 프레임의 requestCapture는 다음 프레임으로 미룬다
    void captureTarget(rhi::RenderTarget target, Image* out);

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
