#include "mealkit/render/Renderer.h"

#include "mealkit/core/Log.h"
#include "mealkit/platform/Image.h"
#include "mealkit/render/DrawList.h"
#include "mealkit/render/ShaderReload.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "g_ForwardPbrPS.h"
#include "g_ForwardToonPS.h"
#include "g_ForwardUnlitPS.h"
#include "g_ForwardVS.h"
#include "g_OutlinePS.h"
#include "g_OutlineVS.h"
#include "g_ForwardSkinVS.h"
#include "g_OutlineSkinVS.h"
#include "g_ShadowPS.h"
#include "g_ShadowSkinVS.h"
#include "g_ShadowVS.h"
#include "g_SkyPS.h"
#include "g_SkyVS.h"
#include "g_ParticlePS.h"
#include "g_ParticleVS.h"
#include "g_BloomDownPS.h"
#include "g_BloomUpPS.h"
#include "g_PostVS.h"
#include "g_UberPS.h"
#include "g_UiPS.h"
#include "g_UiVS.h"

namespace mealkit::render {
namespace {

// shaders/Forward.hlsl FrameConstants와 같은 배치 (업로드 링에 쓰고 셰이더가 Load로 읽는다)
struct FrameConstants {
    f32 viewProj[16];
    f32 cameraPos[4];
    f32 sunDir[4];
    f32 sunColor[4];
    f32 skyColor[4];
    f32 groundColor[4];
    f32 fogColor[4];
    f32 screenTint[4];
    u32 materials[4]; // 머티리얼 표의 링 위치, 개수, 0, 0
    f32 wind[4];      // 수평 방향 x, z, 세기, 화면 시간
    u32 style[4];     // 툰 램프 텍스처 (bindless), 0, 0, 0
    f32 invViewProj[16]; // 스카이박스: 화면 → 월드 방향
    u32 env[4];          // 하늘 큐브, 반사 큐브 (g_cubes 번호), 반사 밉 수, 있음(1)
    f32 envParams[4];    // 세기, cos(회전), sin(회전), 0
    f32 sh[9][4];        // 환경광 SH (rgb, 0)
    u32 probes[4];       // 반사 프로브 표의 링 위치, 개수, 0, 0 (표: 프로브마다 위치 xyz, 큐브 번호 | 밉 수 << 24, 상자 반 크기 xyz, 0)
    u32 lights[4];       // 점광원 표의 링 위치, 개수, 0, 0 (표: 빛마다 위치 xyz, 거리, 색 rgb, 0)
    f32 shadowViewProj[kMaxShadowCascades][16]; // 단계마다 월드 → 그림자 클립 (행 우선)
    f32 shadowForward[4]; // 카메라 앞 방향 xyz (단계 고르기: 앞 방향 거리), 0
    f32 shadowSplits[4];  // 단계 0 끝 거리, 마지막 단계 끝 거리, 단계 0·1 텍셀의 월드 크기 (법선 밀기)
    u32 shadow[4];        // 그림자 맵 텍스처 (bindless), 단계 수 (0 = 그림자 없음), 단계 하나 크기, 0
};
static_assert(sizeof(FrameConstants) == 672);

// 4×4 역행렬 (행 우선, 행 벡터 규칙 그대로). 특이 행렬이면 단위 행렬
void invert(const f32 m[16], f32 out[16]) {
    f32 inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const f32 det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (std::abs(det) < 1e-30f) {
        for (int i = 0; i < 16; ++i) out[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        return;
    }
    for (int i = 0; i < 16; ++i) out[i] = inv[i] / det;
}

// shaders/Forward.hlsl Material (128바이트)
struct MaterialGpu {
    f32 baseColor[4];
    f32 emissive[3];
    f32 alphaCutoff;
    f32 metallic, roughness, normalScale;
    u32 features;    // kMaterial* 비트
    u32 textures[4]; // baseColor, normal, metallicRoughness, emissive (bindless 번호, kNoTexture = 없음)
    f32 outlineColor[3];
    f32 outlineWidth;
    f32 rimColor[3];
    f32 rimStrength;
    f32 dissolveColor[3];
    f32 dissolveEdge;
    f32 windStrength;
    f32 pad[3];
};
static_assert(sizeof(MaterialGpu) == 128);

// 픽셀 셰이더 변형 (render::LightingModel 순서, CMakeLists의 mealkit_shader LIGHTING_MODEL=n)
const rhi::ShaderCode kLightingPS[kLightingModelCount] = {
    {g_ForwardPbrPS, sizeof(g_ForwardPbrPS)}, {g_ForwardToonPS, sizeof(g_ForwardToonPS)}, {g_ForwardUnlitPS, sizeof(g_ForwardUnlitPS)}};
const char* const kLightingDefines[kLightingModelCount] = {"LIGHTING_MODEL=0", "LIGHTING_MODEL=1", "LIGHTING_MODEL=2"};

// 툰 램프 텍스처 너비. 프로젝트 그림은 이 너비로 가장 가까운 픽셀을 골라 줄인다(늘린다): 단계 경계가 흐려지지 않는다
constexpr u32 kToonRampWidth = 256;

struct ForwardPush {
    u32 ring = 0;
    u32 frameOffset = 0;
    u32 instanceOffset = 0;
    u32 debugView = 0;   // render::DebugView (Forward.hlsl PSMain)
    u32 lightOffset = 0; // 인스턴스마다 점광원 번호 8개(8바이트, 0xFF = 끝)의 링 위치 (프레임에 빛이 있을 때만 읽는다)
    u32 paletteOffset = 0; // 스킨 메시: 팔레트(관절마다 48바이트)의 링 위치
    u32 pad[2] = {};
};

struct UiPush {
    f32 screenSize[2] = {};
    u32 ring = 0;
    u32 quadOffset = 0;
    u32 glyphAtlas = 0;
    u32 imageAtlas = 0;
    u32 pad[2] = {};
};

// shaders/Post.hlsl PostPush (48바이트)
struct PostPush {
    u32 ring = 0;
    u32 constOffset = 0;
    u32 source = 0;
    u32 first = 0;
    f32 rect[4] = {};
    f32 texel[2] = {};
    f32 scale[2] = {};
};

// shaders/Post.hlsl PostConstants (업로드 링, 96바이트)
struct PostConstants {
    f32 exposure[4]; // 노출 배율, 블룸 세기, 비네트 세기, 비네트 부드러움
    f32 effects[4];  // 그레인, 색수차, 시간, 0
    u32 textures[4]; // 블룸 사슬 맨 위, LUT, 플래그, 0 (없음 = kNoTexture)
    f32 bloom[4];    // 블룸 색, 문턱
    f32 region[4];   // 장면 영역 uv
    f32 sceneSize[4];
};
static_assert(sizeof(PostConstants) == 96);
constexpr u32 kPostToneMap = 1u, kPostFxaa = 2u, kPostDither = 4u; // Post.hlsl kPost*
constexpr u32 kLutWidth = kLutSize * kLutSize; // 32³을 가로 띠 1024 × 32로 (파랑 조각마다 32칸)

// 색맹 보정 (Fidaner 계열 daltonize): 선형 RGB에서 Machado 2009(심한 정도 1.0)로 그 사람이 보는 색을 흉내 내고,
// 못 보는 차이를 볼 수 있는 채널로 옮긴다. 입력·출력은 0..1 sRGB 값 (LUT 칸)
void daltonize(ColorBlindMode mode, f32 c[3]) {
    static const f32 kSimulate[3][9] = {
        {0.152286f, 1.052583f, -0.204868f, 0.114503f, 0.786281f, 0.099216f, -0.003882f, -0.048116f, 1.051998f},  // 1형 (빨강)
        {0.367322f, 0.860646f, -0.227968f, 0.280085f, 0.672501f, 0.047413f, -0.011820f, 0.042940f, 0.968881f},   // 2형 (초록)
        {1.255528f, -0.076749f, -0.178779f, -0.078411f, 0.930809f, 0.147602f, 0.004733f, 0.691367f, 0.303900f}}; // 3형 (파랑)
    if (mode == ColorBlindMode::Off) return;
    const f32* s = kSimulate[static_cast<u32>(mode) - 1];
    f32 lin[3];
    for (u32 i = 0; i < 3; ++i) lin[i] = c[i] <= 0.04045f ? c[i] / 12.92f : std::pow((c[i] + 0.055f) / 1.055f, 2.4f);
    f32 err[3];
    for (u32 i = 0; i < 3; ++i) err[i] = lin[i] - (s[i * 3] * lin[0] + s[i * 3 + 1] * lin[1] + s[i * 3 + 2] * lin[2]);
    const f32 out[3] = {lin[0], lin[1] + 0.7f * err[0] + err[1], lin[2] + 0.7f * err[0] + err[2]};
    for (u32 i = 0; i < 3; ++i) {
        const f32 v = std::clamp(out[i], 0.0f, 1.0f);
        c[i] = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    }
}

constexpr u32 kRingAlignment = 16;

struct GpuMesh {
    rhi::Buffer vertexBuffer;
    rhi::Buffer indexBuffer;
    u32 indexCount = 0;
    BoundingSphere bounds;
    bool outline = false; // 외곽선 머티리얼이 하나라도 있다: 외곽선 패스로 한 번 더 그린다
    bool skinned = false; // 정점 44바이트 (Vertex + SkinWeights), 스킨 파이프라인으로 그린다
};

// 스킨 파이프라인: 같은 desc에 정점 셰이더(SKINNED=1)와 관절 속성 둘, 정점 간격 44바이트
rhi::PipelineDesc skinnedDesc(rhi::PipelineDesc d, rhi::ShaderCode vs) {
    d.vs = vs;
    d.vertexAttributes.push_back({"JOINTS", rhi::VertexFormat::Uint8x4, sizeof(Vertex)});
    d.vertexAttributes.push_back({"WEIGHTS", rhi::VertexFormat::Unorm8x4, sizeof(Vertex) + 4});
    d.vertexStride = kSkinnedVertexStride;
    return d;
}

} // namespace

struct Renderer::Impl {
    rhi::Device device;
    rhi::Pipeline forward[kLightingModelCount]; // 조명 모델마다 (init이 모두 만든다: 바꿔도 끊기지 않는다)
    rhi::Pipeline outline;                      // 외곽선 패스 (뒤집은 껍질)
    // 스킨 메시 변형 (정점 셰이더만 다르다: SKINNED=1). 보기 모드 둘은 디버그 UI 빌드만
    rhi::Pipeline forwardSkin[kLightingModelCount], outlineSkin, shadowSkin, wireSkin, overdrawSkin;
    rhi::PipelineDesc forwardSkinDesc[kLightingModelCount], outlineSkinDesc, shadowSkinDesc;
    rhi::Pipeline forwardWire, forwardOverdraw; // 에디터 보기 모드 (디버그 UI 빌드만, PBR 셰이더)
    rhi::Pipeline ui;
    rhi::Pipeline particles; // shaders/Particle.hlsl (불투명 3D 다음, 깊이 쓰기 없음)
    rhi::PipelineDesc particleDesc;
    rhi::Pipeline sky; // 스카이박스 (화면 가득 삼각형, 깊이 없이 3D 전에)
    rhi::PipelineDesc skyDesc;
    struct Environment {
        u32 sky = 0, reflect = 0, reflectMips = 0; // g_cubes 번호
        f32 sh[9][3] = {};
    };
    std::vector<Environment> environments; // createEnvironment 번호 - 1
    std::vector<u32> probes;               // createProbe 번호 - 1 → 큐브 번호 | 밉 수 << 24
    std::vector<PointLight> frameLights;   // 이번 프레임 상수에 쓴 점광원 (drawInstances가 인스턴스마다 고른다)
    u32 lightsPerObject = 4;
    std::vector<u8> lightPicks;            // drawInstances 작업 버퍼

    // 태양 그림자: 단계들을 가로로 붙인 깊이 아틀라스 하나 (단계 i는 x = i × size)
    ShadowSettings shadowSettings;
    rhi::RenderTarget shadowTarget;
    u32 shadowTargetSize = 0, shadowTargetCascades = 0;
    rhi::Pipeline shadowPipeline;
    rhi::PipelineDesc shadowDesc;
    u32 shadowCascades = 0;                    // 이번 프레임 (0 = 그리지 않는다)
    Mat4 shadowViewProj[kMaxShadowCascades];
    u64 shadowFrameOffsets[kMaxShadowCascades] = {}; // 단계마다 viewProj만 바꾼 프레임 상수
    bool shadowPending = false;                // 이번 프레임 그림자 맵을 아직 그리지 않았다
    void computeCascades(const FrameParams& p, FrameConstants& fc);

    // 포스트 프로세싱 (shaders/Post.hlsl): HDR 장면 대상 + 블룸 사슬(절반부터 반씩, 처음 쓸 때 만든다).
    // 본 화면은 창 크기(beginFrame이 맞춘다), 오프스크린은 출력 대상(Ldr)마다 같은 크기의 짝
    static constexpr u32 kBloomLevels = 5;
    struct PostChain {
        rhi::RenderTarget scene;
        rhi::RenderTarget bloom[kBloomLevels];
        u32 width = 0, height = 0;
        u32 bloomSize[kBloomLevels][2] = {};
    };
    PostChain mainChain;
    struct Offscreen {
        rhi::RenderTarget output; // createRenderTarget이 준 Ldr 대상 (0 = 이 번호는 다른 대상)
        PostChain chain;
    };
    std::vector<Offscreen> offscreen; // RenderTarget::id로
    rhi::Pipeline bloomDown, bloomUp, uber;
    rhi::PipelineDesc bloomDownDesc, bloomUpDesc, uberDesc;
    PostOptions postOptions;
    struct Lut {
        std::vector<f32> rgb; // kLutSize³ × 3 (색맹 보정을 바꾸면 다시 합친다)
        rhi::Texture texture; // kLutWidth × kLutSize RGBA8
    };
    std::vector<Lut> luts; // 0 = 항등 (색맹 보정만 할 때)
    void uploadLut(u32 index);
    void resizeChain(PostChain& chain, u32 width, u32 height); // 프레임 밖에서 (옛 대상은 지운다)
    void ensureBloom(PostChain& chain);
    // 장면 대상(chain.scene, endTarget까지 끝난)을 output(없으면 백버퍼의 rect 영역)으로. timed: GPU 패스 Post로 잰다
    void runPost(PostChain& chain, const PostParams& post, const f32 rect[4], rhi::RenderTarget output, bool timed, f32 time);
    bool sceneResolved = false; // 본 화면을 포스트로 백버퍼에 냈다 (이 뒤로 본 화면 3D 없음)
    void resolveScene();        // ensureScene 다음에

    // 본 화면 3D 대상은 처음 그릴 때 묶는다 (그림자 패스가 먼저 다른 대상에 그릴 수 있게)
    bool sceneBound = false;
    f32 sceneClear[4] = {};
    f32 sceneViewport[4] = {};
    bool sceneSky = false;
    FrameParams sceneParams; // drawSky가 보는 환경 (포인터 필드는 쓰지 않는다)
    void ensureScene();
    rhi::RenderTarget captureSource;       // captureTarget (이번 endFrame)
    Image* captureOut = nullptr;
    void drawSky(const FrameParams& p); // 프레임 상수를 쓴 뒤, 3D 전에 (환경이 없으면 아무것도 하지 않는다)
    LightingModel lighting = LightingModel::Pbr;
    DebugView debugView = DebugView::None;
    // 셰이더 핫 리로드가 바이트코드만 바꿔 다시 만든다. forwardDesc[n]의 VS는 모두 같다
    rhi::PipelineDesc forwardDesc[kLightingModelCount], outlineDesc, uiDesc;
    ShaderReloader shaderReload;
    std::vector<GpuMesh> meshes;
    std::vector<MaterialGpu> materials; // 0 = 기본 흰색. 프레임마다 업로드 링에 쓴다
    std::vector<rhi::Texture> textures;
    rhi::Texture toonRamp; // kToonRampWidth × 1, setToonRamp가 내용을 바꾼다

    // 2D UI (shaders/Ui.hlsl): 글자 SDF 아틀라스 + 이미지 아틀라스
    rhi::Texture uiAtlas[2];
    struct TextureUpload {
        rhi::Texture texture;
        u32 bpp;
        u32 x, y, w, h;
        std::vector<u8> pixels;
    };
    std::vector<TextureUpload> uploads; // 아직 GPU에 올리지 않은 영역 (UI 아틀라스, 툰 램프. 차례대로)

    u64 frameOffset = 0; // 이번 프레임 FrameConstants의 링 위치
    u64 materialOffset = 0; // 이번 프레임 머티리얼 표의 링 위치 (본 화면과 오프스크린이 같이)
    // 업로드 링에 쓰고 위치를 offset에. withShadows: 본 화면 (태양 그림자 단계를 정해 넣는다)
    void writeFrameConstants(const FrameParams& p, u64* offset, bool withShadows);
    bool vsync = true;
    bool inFrame = false;
    bool uploadOverflowWarned = false;
    RenderStats stats;
    Frustum frustum;
    i32 openPass = -1; // 지금 GPU 시간을 재는 패스 (GpuPass, 없으면 -1)
    std::string capturePath;

    rhi::PipelineDesc wireDesc() const {
        rhi::PipelineDesc d = forwardDesc[0];
        d.wireframe = true;
        d.cullBack = false;
        return d;
    }
    rhi::PipelineDesc overdrawDesc() const {
        rhi::PipelineDesc d = forwardDesc[0];
        d.additiveBlend = true;
        d.depthTest = false;
        d.cullBack = false;
        return d;
    }

    // 기다리던 아틀라스 영역을 이번 프레임 업로드 링으로 복사한다 (자리가 없으면 다음 프레임에)
    void beginPass(GpuPass pass) {
        endPass();
        openPass = static_cast<i32>(pass);
        device.cmd().beginTimer(static_cast<u32>(pass));
    }
    void endPass() {
        if (openPass >= 0) device.cmd().endTimer(static_cast<u32>(openPass));
        openPass = -1;
    }

    void flushUploads() {
        rhi::CommandList& cmd = device.cmd();
        size_t done = 0;
        for (; done < uploads.size(); ++done) {
            const TextureUpload& u = uploads[done];
            const u32 pitch = device.copyPitch(u.w, u.bpp);
            u64 offset = 0;
            u8* dst = device.allocUpload(static_cast<u64>(pitch) * u.h, device.copyAlignment(), &offset);
            if (!dst) break;
            for (u32 row = 0; row < u.h; ++row)
                std::memcpy(dst + static_cast<size_t>(row) * pitch, u.pixels.data() + static_cast<size_t>(row) * u.w * u.bpp,
                            static_cast<size_t>(u.w) * u.bpp);
            cmd.textureBarrier(u.texture, rhi::TextureState::ShaderRead, rhi::TextureState::CopyDst);
            cmd.copyToTexture(u.texture, u.x, u.y, u.w, u.h, offset, pitch);
            cmd.textureBarrier(u.texture, rhi::TextureState::CopyDst, rhi::TextureState::ShaderRead);
        }
        uploads.erase(uploads.begin(), uploads.begin() + static_cast<std::ptrdiff_t>(done));
    }

    rhi::Pipeline& scenePipeline(bool skinned = false) {
        if (skinned)
            return debugView == DebugView::Wireframe ? wireSkin
                   : debugView == DebugView::Overdraw ? overdrawSkin
                   : debugView == DebugView::None     ? forwardSkin[static_cast<u32>(lighting)]
                                                      : forwardSkin[0];
        return debugView == DebugView::Wireframe ? forwardWire
               : debugView == DebugView::Overdraw ? forwardOverdraw
               : debugView == DebugView::None     ? forward[static_cast<u32>(lighting)]
                                                  : forward[0]; // 보기 모드는 조명과 무관한 색 (PBR 셰이더 안)
    }
    void drawMesh(MeshId meshId, const InstanceData* instances, u32 count, u32 paletteOffset); // drawInstances·drawSkinned

    void warnOverflow(u64 bytes) {
        if (uploadOverflowWarned) return;
        MEALKIT_WARN("Upload ring full (%llu bytes requested); raise kUploadBytesPerFrame",
                     static_cast<unsigned long long>(bytes));
        uploadOverflowWarned = true;
    }
};

Renderer::~Renderer() { shutdown(); }

bool Renderer::init(const RendererDesc& desc) {
    m = new Impl();
    m->vsync = desc.vsync;
    if (!m->device.init({desc.window, desc.width, desc.height, desc.debugLayer, desc.gpuCapture})) return false;

    // 스타일 프리셋: 조명 모델 셋 + 외곽선 파이프라인을 여기서 모두 만든다 (불러오는 중 한 번, 플레이 중 처음 보는 조합이 없다)
    rhi::PipelineDesc forward;
    forward.vs = {g_ForwardVS, sizeof(g_ForwardVS)};
    forward.vertexAttributes = {{"POSITION", rhi::VertexFormat::Float3, 0},
                                {"NORMAL", rhi::VertexFormat::Float3, 12},
                                {"COLOR", rhi::VertexFormat::Unorm8x4, 24},
                                {"TEXCOORD", rhi::VertexFormat::Half2, 28},
                                {"MATERIAL", rhi::VertexFormat::Uint16x2, 32}};
    forward.vertexStride = sizeof(Vertex);
    forward.target = rhi::Target::Scene;
    for (u32 i = 0; i < kLightingModelCount; ++i) {
        m->forwardDesc[i] = forward;
        m->forwardDesc[i].ps = kLightingPS[i];
        m->forward[i] = m->device.createPipeline(m->forwardDesc[i]);
    }
    m->outlineDesc = forward;
    m->outlineDesc.vs = {g_OutlineVS, sizeof(g_OutlineVS)};
    m->outlineDesc.ps = {g_OutlinePS, sizeof(g_OutlinePS)};
    m->outlineDesc.cullFront = true; // 부푼 껍질의 뒷면만: 앞의 물체를 가리지 않고 둘레에 테두리로 남는다
    m->outline = m->device.createPipeline(m->outlineDesc);
#if MEALKIT_DEBUG_UI
    m->forwardWire = m->device.createPipeline(m->wireDesc());
    m->forwardOverdraw = m->device.createPipeline(m->overdrawDesc());
#endif

    rhi::PipelineDesc ui;
    ui.vs = {g_UiVS, sizeof(g_UiVS)};
    ui.ps = {g_UiPS, sizeof(g_UiPS)};
    ui.target = rhi::Target::Overlay;
    ui.cullBack = false;
    ui.premultipliedBlend = true; // 미리 곱한 알파
    m->ui = m->device.createPipeline(ui);
    m->uiDesc = ui;
    // 개발 빌드: engine/shaders가 바뀌면 다시 컴파일해 바꿔 끼운다 (출시 빌드와 엔진 폴더가 없는 PC에서는 아무것도 하지 않는다)
    for (u32 i = 0; i < kLightingModelCount; ++i) {
        m->shaderReload.add(
            "Forward.hlsl",
            [this, i](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
                rhi::PipelineDesc d = m->forwardDesc[i];
                d.vs = vs;
                d.ps = ps;
                m->forwardDesc[i] = d; // 바이트코드는 ShaderReloader가 갖고 있다
#if MEALKIT_DEBUG_UI
                if (i == 0) { // 보기 모드 파이프라인은 PBR 셰이더
                    m->device.recreatePipeline(m->forwardWire, m->wireDesc());
                    m->device.recreatePipeline(m->forwardOverdraw, m->overdrawDesc());
                }
#endif
                return m->device.recreatePipeline(m->forward[i], d);
            },
            "VSMain", "PSMain", kLightingDefines[i]);
    }
    m->shaderReload.add(
        "Forward.hlsl",
        [this](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
            rhi::PipelineDesc d = m->outlineDesc;
            d.vs = vs;
            d.ps = ps;
            m->outlineDesc = d;
            return m->device.recreatePipeline(m->outline, d);
        },
        "VSOutline", "PSOutline");
    rhi::PipelineDesc sky;
    sky.vs = {g_SkyVS, sizeof(g_SkyVS)};
    sky.ps = {g_SkyPS, sizeof(g_SkyPS)};
    sky.target = rhi::Target::Scene;
    sky.cullBack = false;
    sky.depthTest = false; // 먼저 그리고 3D가 덮는다 (깊이도 쓰지 않는다)
    m->skyDesc = sky;
    m->sky = m->device.createPipeline(sky);
    m->shaderReload.add(
        "Forward.hlsl",
        [this](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
            rhi::PipelineDesc d = m->skyDesc;
            d.vs = vs;
            d.ps = ps;
            m->skyDesc = d;
            return m->device.recreatePipeline(m->sky, d);
        },
        "VSSky", "PSSky");
    rhi::PipelineDesc shadow = forward; // 그림자: 같은 정점 형식, 깊이만 (알파 마스크·디졸브만 픽셀 셰이더에서 버린다)
    shadow.vs = {g_ShadowVS, sizeof(g_ShadowVS)};
    shadow.ps = {g_ShadowPS, sizeof(g_ShadowPS)};
    shadow.target = rhi::Target::Shadow;
    shadow.cullBack = false; // 양면: 얇은 물체와 열린 모델도 그림자가 빈틈없다 (빛 새는 것은 법선 밀기가 막는다)
    m->shadowDesc = shadow;
    m->shadowPipeline = m->device.createPipeline(shadow);
    const rhi::ShaderCode skinVS{g_ForwardSkinVS, sizeof(g_ForwardSkinVS)};
    for (u32 i = 0; i < kLightingModelCount; ++i) {
        m->forwardSkinDesc[i] = skinnedDesc(m->forwardDesc[i], skinVS);
        m->forwardSkin[i] = m->device.createPipeline(m->forwardSkinDesc[i]);
    }
    m->outlineSkinDesc = skinnedDesc(m->outlineDesc, {g_OutlineSkinVS, sizeof(g_OutlineSkinVS)});
    m->outlineSkin = m->device.createPipeline(m->outlineSkinDesc);
    m->shadowSkinDesc = skinnedDesc(m->shadowDesc, {g_ShadowSkinVS, sizeof(g_ShadowSkinVS)});
    m->shadowSkin = m->device.createPipeline(m->shadowSkinDesc);
#if MEALKIT_DEBUG_UI
    m->wireSkin = m->device.createPipeline(skinnedDesc(m->wireDesc(), skinVS));
    m->overdrawSkin = m->device.createPipeline(skinnedDesc(m->overdrawDesc(), skinVS));
#endif
    // 셰이더 핫 리로드: 스킨 변형 (정의 두 개는 한 줄로: "-D"를 사이에 둔다)
    static const char* const kSkinDefines[kLightingModelCount] = {"LIGHTING_MODEL=0 -D SKINNED=1", "LIGHTING_MODEL=1 -D SKINNED=1",
                                                                  "LIGHTING_MODEL=2 -D SKINNED=1"};
    for (u32 i = 0; i < kLightingModelCount; ++i)
        m->shaderReload.add(
            "Forward.hlsl",
            [this, i](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
                rhi::PipelineDesc d = m->forwardSkinDesc[i];
                d.vs = vs;
                d.ps = ps;
                m->forwardSkinDesc[i] = d;
                return m->device.recreatePipeline(m->forwardSkin[i], d);
            },
            "VSMain", "PSMain", kSkinDefines[i]);
    m->shaderReload.add(
        "Forward.hlsl",
        [this](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
            rhi::PipelineDesc d = m->outlineSkinDesc;
            d.vs = vs;
            d.ps = ps;
            m->outlineSkinDesc = d;
            return m->device.recreatePipeline(m->outlineSkin, d);
        },
        "VSOutline", "PSOutline", "SKINNED=1");
    m->shaderReload.add(
        "Forward.hlsl",
        [this](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
            rhi::PipelineDesc d = m->shadowSkinDesc;
            d.vs = vs;
            d.ps = ps;
            m->shadowSkinDesc = d;
            return m->device.recreatePipeline(m->shadowSkin, d);
        },
        "VSShadow", "PSShadow", "SKINNED=1");
    m->shaderReload.add(
        "Forward.hlsl",
        [this](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
            rhi::PipelineDesc d = m->shadowDesc;
            d.vs = vs;
            d.ps = ps;
            m->shadowDesc = d;
            return m->device.recreatePipeline(m->shadowPipeline, d);
        },
        "VSShadow", "PSShadow");
    rhi::PipelineDesc particles;
    particles.vs = {g_ParticleVS, sizeof(g_ParticleVS)};
    particles.ps = {g_ParticlePS, sizeof(g_ParticlePS)};
    particles.target = rhi::Target::Scene;
    particles.cullBack = false;
    particles.premultipliedBlend = true; // 더하기 파티클은 알파 0으로 낸다
    particles.depthWrite = false;
    m->particleDesc = particles;
    m->particles = m->device.createPipeline(particles);
    m->shaderReload.add("Particle.hlsl", [this](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
        rhi::PipelineDesc d = m->particleDesc;
        d.vs = vs;
        d.ps = ps;
        return m->device.recreatePipeline(m->particles, d);
    });
    m->shaderReload.add("Ui.hlsl", [this](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
        rhi::PipelineDesc d = m->uiDesc;
        d.vs = vs;
        d.ps = ps;
        return m->device.recreatePipeline(m->ui, d);
    });
    // 포스트: 화면 가득 삼각형 (정점 버퍼·깊이 없음). 블룸 사슬은 HDR, 우버 패스는 백버퍼·Ldr 대상
    rhi::PipelineDesc post;
    post.vs = {g_PostVS, sizeof(g_PostVS)};
    post.cullBack = false;
    post.depthTest = false;
    post.target = rhi::Target::Hdr;
    m->bloomDownDesc = post;
    m->bloomDownDesc.ps = {g_BloomDownPS, sizeof(g_BloomDownPS)};
    m->bloomUpDesc = post;
    m->bloomUpDesc.ps = {g_BloomUpPS, sizeof(g_BloomUpPS)};
    m->bloomUpDesc.additiveBlend = true; // 그 단계의 내려온 값 위에 쌓는다
    m->uberDesc = post;
    m->uberDesc.ps = {g_UberPS, sizeof(g_UberPS)};
    m->uberDesc.target = rhi::Target::Overlay;
    m->bloomDown = m->device.createPipeline(m->bloomDownDesc);
    m->bloomUp = m->device.createPipeline(m->bloomUpDesc);
    m->uber = m->device.createPipeline(m->uberDesc);
    const struct {
        rhi::Pipeline Impl::*pipeline;
        rhi::PipelineDesc Impl::*desc;
        const char* entry;
    } postShaders[] = {{&Impl::bloomDown, &Impl::bloomDownDesc, "PSBloomDown"},
                       {&Impl::bloomUp, &Impl::bloomUpDesc, "PSBloomUp"},
                       {&Impl::uber, &Impl::uberDesc, "PSUber"}};
    for (const auto& s : postShaders)
        m->shaderReload.add(
            "Post.hlsl",
            [this, s](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
                rhi::PipelineDesc& d = m->*s.desc;
                d.vs = vs;
                d.ps = ps;
                return m->device.recreatePipeline(m->*s.pipeline, d);
            },
            "VSMain", s.entry);

    createMaterial({}); // 0번: 기본 흰색 (머티리얼 없는 메시, 상자)
    std::vector<f32> identity(static_cast<size_t>(kLutSize) * kLutSize * kLutSize * 3);
    for (u32 i = 0; i < kLutSize * kLutSize * kLutSize; ++i) {
        identity[i * 3 + 0] = static_cast<f32>(i % kLutSize) / (kLutSize - 1);
        identity[i * 3 + 1] = static_cast<f32>(i / kLutSize % kLutSize) / (kLutSize - 1);
        identity[i * 3 + 2] = static_cast<f32>(i / (kLutSize * kLutSize)) / (kLutSize - 1);
    }
    createLut(identity.data()); // 0번: 항등 (색맹 보정만 할 때)
    m->uiAtlas[0] = m->device.createTexture(kUiAtlasSize, kUiAtlasSize, rhi::Format::R8Unorm);
    m->uiAtlas[1] = m->device.createTexture(kUiAtlasSize, kUiAtlasSize, rhi::Format::Rgba8Unorm);
    m->toonRamp = m->device.createTexture(kToonRampWidth, 1, rhi::Format::Rgba8Unorm);
    setToonRamp(nullptr);
    return true;
}

void Renderer::setShadows(const ShadowSettings& shadows) {
    if (!m) return;
    m->shadowSettings = shadows;
    m->shadowSettings.cascades = std::clamp<u32>(shadows.cascades, 1, kMaxShadowCascades);
    m->shadowSettings.size = std::clamp<u32>(shadows.size, 512, 4096);
    m->shadowSettings.distance = std::max(shadows.distance, 1.0f);
}

const ShadowSettings& Renderer::shadows() const {
    static const ShadowSettings none{};
    return m ? m->shadowSettings : none;
}

void Renderer::setLightsPerObject(u32 count) {
    if (m) m->lightsPerObject = std::clamp<u32>(count, 1, kMaxLightsPerObject);
}

void Renderer::setLightingModel(LightingModel model) {
    if (m && static_cast<u32>(model) < kLightingModelCount) m->lighting = model;
}

LightingModel Renderer::lightingModel() const { return m ? m->lighting : LightingModel::Pbr; }

void Renderer::setToonRamp(const Image* ramp) {
    if (!m) return;
    Impl::TextureUpload u;
    u.texture = m->toonRamp;
    u.bpp = 4;
    u.x = u.y = 0;
    u.w = kToonRampWidth;
    u.h = 1;
    u.pixels.resize(kToonRampWidth * 4);
    for (u32 x = 0; x < kToonRampWidth; ++x) {
        u8* px = &u.pixels[x * 4];
        if (ramp && ramp->width && ramp->rgba.size() >= static_cast<size_t>(ramp->width) * 4) {
            const u32 sx = std::min(ramp->width - 1, x * ramp->width / kToonRampWidth); // 첫 줄, 가장 가까운 픽셀
            std::memcpy(px, &ramp->rgba[sx * 4], 4);
        } else {
            // 엔진 기본 3단계 (sRGB, 셰이더가 제곱해 선형 0.35 / 0.72 / 1.0): 그늘 → 중간 → 밝은 면
            const f32 t = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(kToonRampWidth);
            const u8 v = t < 0.42f ? 151 : t < 0.6f ? 216 : 255;
            px[0] = px[1] = px[2] = v;
            px[3] = 255;
        }
    }
    m->uploads.push_back(std::move(u));
}

void Renderer::shutdown() {
    if (!m) return;
    m->device.shutdown();
    delete m;
    m = nullptr;
}

void Renderer::resize(u32 width, u32 height) {
    if (m) m->device.resize(width, height);
}

u32 Renderer::createMaterial(const MaterialData& d) {
    MaterialGpu g{};
    std::memcpy(g.baseColor, d.baseColor, sizeof(g.baseColor));
    std::memcpy(g.emissive, d.emissive, sizeof(g.emissive));
    g.alphaCutoff = d.alphaCutoff;
    g.metallic = d.metallic;
    g.roughness = d.roughness;
    g.normalScale = d.normalScale;
    const u32 textures[4] = {d.baseColorTexture, d.normalTexture, d.metallicRoughnessTexture, d.emissiveTexture};
    std::memcpy(g.textures, textures, sizeof(textures));
    g.features = d.features;
    std::memcpy(g.outlineColor, d.outlineColor, sizeof(g.outlineColor));
    g.outlineWidth = d.outlineWidth;
    std::memcpy(g.rimColor, d.rimColor, sizeof(g.rimColor));
    g.rimStrength = d.rimStrength;
    std::memcpy(g.dissolveColor, d.dissolveColor, sizeof(g.dissolveColor));
    g.dissolveEdge = d.dissolveEdge;
    g.windStrength = d.windStrength;
    if (m->materials.size() >= 0xFFFF) { // 정점의 material은 16비트
        MEALKIT_WARN("createMaterial: too many materials, using the default");
        return 0;
    }
    m->materials.push_back(g);
    return static_cast<u32>(m->materials.size() - 1);
}

u32 Renderer::createTexture(const rhi::TextureInit& init) {
    const rhi::Texture t = m->device.createTexture(init);
    m->textures.push_back(t);
    return m->device.textureIndex(t);
}

MeshId Renderer::createMesh(const MeshData& data) {
    if (data.vertices.empty() || data.indices.empty()) {
        MEALKIT_WARN("createMesh: empty mesh");
        return 0;
    }
    GpuMesh mesh;
    mesh.skinned = data.skin.size() == data.vertices.size();
    if (mesh.skinned) { // Vertex 뒤에 SkinWeights를 붙인 44바이트 정점
        std::vector<u8> interleaved(data.vertices.size() * kSkinnedVertexStride);
        for (size_t i = 0; i < data.vertices.size(); ++i) {
            std::memcpy(&interleaved[i * kSkinnedVertexStride], &data.vertices[i], sizeof(Vertex));
            std::memcpy(&interleaved[i * kSkinnedVertexStride + sizeof(Vertex)], &data.skin[i], sizeof(SkinWeights));
        }
        mesh.vertexBuffer = m->device.createBuffer(rhi::BufferUsage::Vertex, interleaved.data(), interleaved.size());
    } else {
        mesh.vertexBuffer =
            m->device.createBuffer(rhi::BufferUsage::Vertex, data.vertices.data(), data.vertices.size() * sizeof(Vertex));
    }
    // 정점이 적으면 16비트 인덱스로 줄여 메모리 대역폭을 아낀다 (내장 그래픽에서 중요)
    if (data.vertices.size() <= 0xFFFF) {
        const std::vector<u16> shortIndices(data.indices.begin(), data.indices.end());
        mesh.indexBuffer = m->device.createBuffer(rhi::BufferUsage::Index16, shortIndices.data(),
                                                  shortIndices.size() * sizeof(u16));
    } else {
        mesh.indexBuffer =
            m->device.createBuffer(rhi::BufferUsage::Index32, data.indices.data(), data.indices.size() * sizeof(u32));
    }
    mesh.indexCount = static_cast<u32>(data.indices.size());
    std::vector<Vec3> points(data.vertices.size());
    for (size_t i = 0; i < points.size(); ++i) points[i] = data.vertices[i].position;
    mesh.bounds = boundingSphere(points.data(), points.size());
    for (const Vertex& v : data.vertices)
        if (v.material < m->materials.size() && (m->materials[v.material].features & kMaterialOutline)) {
            mesh.outline = true;
            break;
        }
    m->meshes.push_back(mesh);
    return static_cast<MeshId>(m->meshes.size() - 1);
}

bool Renderer::beginFrame(const FrameParams& p) {
    if (m) m->shaderReload.update(); // 프레임 밖에서 (파이프라인을 바꿀 때 GPU를 기다린다)
    if (m && width() && height() && (m->mainChain.width != width() || m->mainChain.height != height()))
        m->resizeChain(m->mainChain, width(), height()); // HDR 장면 대상은 창 크기 (지우기는 프레임 밖에서)
    if (!m || !m->mainChain.scene || !m->device.beginFrame()) return false;
    rhi::CommandList& cmd = m->device.cmd();

    const Vec3 sky = lerp(p.fogColor, p.tintColor, std::clamp(p.tintAmount, 0.0f, 1.0f)); // 배경도 번쩍임을 받는다
    m->debugView = supportsDebugView(p.debugView) ? p.debugView : DebugView::None;
    const bool overdraw = m->debugView == DebugView::Overdraw;
    const f32 clear[4] = {overdraw ? 0.0f : sky.x, overdraw ? 0.0f : sky.y, overdraw ? 0.0f : sky.z, 1.0f};
    m->flushUploads(); // 툰 램프·UI 아틀라스 (복사는 그리기 대상을 묶기 전에)
    std::memcpy(m->sceneClear, clear, sizeof(clear));

    const f32 fullW = static_cast<f32>(width()), fullH = static_cast<f32>(height());
    f32 vx = 0.0f, vy = 0.0f, vw = fullW, vh = fullH;
    if (p.viewport.width >= 1.0f && p.viewport.height >= 1.0f) {
        vx = std::clamp(p.viewport.x, 0.0f, fullW - 1.0f);
        vy = std::clamp(p.viewport.y, 0.0f, fullH - 1.0f);
        vw = std::min(p.viewport.width, fullW - vx);
        vh = std::min(p.viewport.height, fullH - vy);
    }
    const f32 viewport[4] = {vx, vy, vw, vh};
    std::memcpy(m->sceneViewport, viewport, sizeof(viewport));
    (void)cmd;

    m->materialOffset = 0;
    const u64 materialBytes = m->materials.size() * sizeof(MaterialGpu);
    if (u8* mat = m->device.allocUpload(materialBytes, kRingAlignment, &m->materialOffset)) std::memcpy(mat, m->materials.data(), materialBytes);
    else m->warnOverflow(materialBytes);
    m->shadowCascades = 0;
    m->writeFrameConstants(p, &m->frameOffset, true); // 그림자 단계도 정한다 (shadowCascades)
    if (m->shadowCascades) { // 단계마다 viewProj만 바꾼 프레임 상수 (그림자 패스의 정점 셰이더)
        FrameParams sp = p;
        sp.lights = nullptr;
        sp.lightCount = 0;
        sp.probes = nullptr;
        sp.probeCount = 0;
        const std::vector<PointLight> frameLights = m->frameLights;
        for (u32 c = 0; c < m->shadowCascades; ++c) {
            sp.viewProj = m->shadowViewProj[c];
            m->writeFrameConstants(sp, &m->shadowFrameOffsets[c], false);
        }
        m->frameLights = frameLights;
    }
    m->shadowPending = m->shadowCascades != 0;
    m->sceneBound = false;
    m->sceneResolved = false;
    m->sceneSky = !overdraw;
    m->sceneParams = p;
    m->sceneParams.probes = nullptr;
    m->sceneParams.lights = nullptr;

    m->stats = {};
    m->frustum = Frustum::fromViewProj(p.viewProj);
    const rhi::GpuTimes& gpu = m->device.gpuTimes();
    m->stats.gpuValid = gpu.valid;
    m->stats.gpuFrameMs = gpu.frameMs;
    for (u32 i = 0; i < kGpuPassCount; ++i) m->stats.gpuPassMs[i] = gpu.timerMs[i];
    m->inFrame = true;
    return true;
}

void Renderer::Impl::ensureScene() {
    if (sceneBound) return;
    sceneBound = true;
    shadowPending = false; // 본 화면을 묶은 뒤에는 그림자 맵을 그리지 않는다 (이번 프레임은 비었거나 그린 것)
    rhi::CommandList& cmd = device.cmd();
    beginPass(GpuPass::Opaque);
    cmd.beginTarget(mainChain.scene, sceneClear, 0.0f); // HDR 장면 대상, reversed-Z: 먼 쪽이 0
    cmd.setViewport(sceneViewport[0], sceneViewport[1], sceneViewport[2], sceneViewport[3]);
    if (sceneSky) drawSky(sceneParams);
    cmd.setPipeline(scenePipeline());
}

// 4×4 행렬로 점 하나 (행 벡터 규칙, w로 나눈다)
static Vec3 transformPoint(const f32 m[16], Vec3 v) {
    const f32 x = v.x * m[0] + v.y * m[4] + v.z * m[8] + m[12];
    const f32 y = v.x * m[1] + v.y * m[5] + v.z * m[9] + m[13];
    const f32 z = v.x * m[2] + v.y * m[6] + v.z * m[10] + m[14];
    const f32 w = v.x * m[3] + v.y * m[7] + v.z * m[11] + m[15];
    return Vec3{x, y, z} * (1.0f / w);
}

// 단계 나누기 (docs/DECISIONS.md 2026-10-27 그림자): 카메라 조각마다 감싸는 구 → 빛 방향 정사영. 구 반지름과 중심을 텍셀 단위로
// 맞춰 카메라가 돌거나 움직여도 그림자 가장자리가 떨리지 않는다. 빛 쪽으로 100 m를 더 잡아 화면 밖 물체의 그림자도 받는다
void Renderer::Impl::computeCascades(const FrameParams& p, FrameConstants& fc) {
    const ShadowSettings& s = shadowSettings;
    if (!s.enabled || debugView != DebugView::None || !shadowPipeline) return;
    const u32 cascades = s.cascades, size = s.size;
    if (!shadowTarget || shadowTargetSize != size || shadowTargetCascades != cascades) {
        shadowTarget = device.createDepthTarget(size * cascades, size);
        shadowTargetSize = size;
        shadowTargetCascades = cascades;
    }
    f32 inv[16];
    invert(&p.viewProj.m[0][0], inv);
    const Vec3 eye = p.cameraPos;
    const Vec3 ahead = normalize(transformPoint(inv, {0.0f, 0.0f, 1.0f}) - eye); // 가까운 면 가운데 (reversed-Z)
    Vec3 rays[4];
    const f32 corners[4][2] = {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    for (u32 i = 0; i < 4; ++i) {
        const Vec3 d = normalize(transformPoint(inv, {corners[i][0], corners[i][1], 1.0f}) - eye);
        rays[i] = d * (1.0f / std::max(dot(d, ahead), 1e-3f)); // 앞 방향 거리 1당 이동
    }
    const f32 splits[kMaxShadowCascades + 1] = {0.0f, cascades == 1 ? s.distance : s.distance * 0.25f, s.distance};
    const Vec3 toSun = normalize(p.sunDirection);
    const Vec3 up = std::abs(toSun.y) > 0.99f ? Vec3{0, 0, 1} : Vec3{0, 1, 0};
    const Mat4 lightView = lookAtLH({}, toSun * -1.0f, up); // 원점에서 빛 방향을 본다 (위치는 정사영 상자가 정한다)
    for (u32 c = 0; c < cascades; ++c) {
        Vec3 points[8];
        for (u32 i = 0; i < 4; ++i) {
            points[i] = eye + rays[i] * std::max(splits[c], 0.1f);
            points[4 + i] = eye + rays[i] * splits[c + 1];
        }
        Vec3 center{};
        for (const Vec3& q : points) center = center + q * 0.125f;
        f32 radius = 0.0f;
        for (const Vec3& q : points) radius = std::max(radius, length(q - center));
        radius = std::ceil(radius * 16.0f) / 16.0f;
        const f32 texel = 2.0f * radius / static_cast<f32>(size);
        Vec3 c0 = transformPoint(&lightView.m[0][0], center);
        c0.x = std::floor(c0.x / texel) * texel;
        c0.y = std::floor(c0.y / texel) * texel;
        const f32 zNear = c0.z - radius - 100.0f, zFar = c0.z + radius;
        Mat4 ortho; // 정사영 (reversed-Z: 가까운 면 1, 먼 면 0)
        ortho.m[0][0] = 1.0f / radius;
        ortho.m[1][1] = 1.0f / radius;
        ortho.m[2][2] = -1.0f / (zFar - zNear);
        ortho.m[3][0] = -c0.x / radius;
        ortho.m[3][1] = -c0.y / radius;
        ortho.m[3][2] = zFar / (zFar - zNear);
        ortho.m[3][3] = 1.0f;
        shadowViewProj[c] = lightView * ortho;
        std::memcpy(fc.shadowViewProj[c], &shadowViewProj[c].m[0][0], sizeof(fc.shadowViewProj[c]));
        fc.shadowSplits[2 + c] = texel;
    }
    const f32 fwd[4] = {ahead.x, ahead.y, ahead.z, 0.0f};
    std::memcpy(fc.shadowForward, fwd, sizeof(fwd));
    fc.shadowSplits[0] = splits[1];
    fc.shadowSplits[1] = s.distance;
    fc.shadow[0] = device.textureIndex(device.renderTargetTexture(shadowTarget));
    fc.shadow[1] = cascades;
    fc.shadow[2] = size;
    shadowCascades = cascades;
}

void Renderer::drawShadows(const DrawList& list) {
    if (!m || !m->inFrame || !m->shadowPending || m->sceneBound) return;
    m->shadowPending = false;
    rhi::CommandList& cmd = m->device.cmd();
    m->beginPass(GpuPass::Shadow);
    const f32 none[4] = {};
    cmd.beginTarget(m->shadowTarget, none, 0.0f);
    cmd.setPipeline(m->shadowPipeline);
    const f32 size = static_cast<f32>(m->shadowTargetSize);
    std::vector<InstanceData> visible;
    for (u32 c = 0; c < m->shadowCascades; ++c) {
        cmd.setViewport(size * static_cast<f32>(c), 0.0f, size, size);
        const Frustum frustum = Frustum::fromViewProj(m->shadowViewProj[c]);
        const auto& buckets = list.buckets();
        for (size_t mesh = 0; mesh < buckets.size() && mesh < m->meshes.size(); ++mesh) {
            if (buckets[mesh].empty()) continue;
            const BoundingSphere b = m->meshes[mesh].bounds;
            visible.clear();
            for (const InstanceData& inst : buckets[mesh]) {
                const Vec3 s = inst.scale;
                const f32 scale = std::max({std::abs(s.x), std::abs(s.y), std::abs(s.z)});
                const Vec3 center = inst.position + rotate(inst.rotation, Vec3{b.center.x * s.x, b.center.y * s.y, b.center.z * s.z});
                if (frustum.sphereVisible(center, b.radius * scale)) visible.push_back(inst);
            }
            if (visible.empty()) continue;
            const u64 bytes = visible.size() * sizeof(InstanceData);
            u64 offset = 0;
            u8* cpu = m->device.allocUpload(bytes, kRingAlignment, &offset);
            if (!cpu) {
                m->warnOverflow(bytes);
                continue;
            }
            std::memcpy(cpu, visible.data(), bytes);
            const GpuMesh& gm = m->meshes[mesh];
            if (gm.skinned) continue; // 스킨 메시는 아래에서 팔레트와 함께
            const ForwardPush push{m->device.uploadBufferIndex(), static_cast<u32>(m->shadowFrameOffsets[c]), static_cast<u32>(offset), 0, 0};
            cmd.pushConstants(&push, sizeof(push));
            cmd.setVertexBuffer(gm.vertexBuffer, sizeof(Vertex));
            cmd.setIndexBuffer(gm.indexBuffer);
            cmd.drawIndexed(gm.indexCount, static_cast<u32>(visible.size()));
            m->stats.drawCalls += 1;
        }
        // 스킨 메시: 캐릭터마다 팔레트를 올리고 하나씩
        bool skinBound = false;
        for (const DrawList::SkinnedItem& item : list.skinned()) {
            if (item.mesh >= m->meshes.size() || !m->meshes[item.mesh].skinned) continue;
            const GpuMesh& gm = m->meshes[item.mesh];
            const InstanceData& inst = item.instance;
            const Vec3 s = inst.scale;
            const f32 scale = std::max({std::abs(s.x), std::abs(s.y), std::abs(s.z)}) * DrawList::kSkinnedBoundsScale;
            const Vec3 center = inst.position + rotate(inst.rotation, Vec3{gm.bounds.center.x * s.x, gm.bounds.center.y * s.y, gm.bounds.center.z * s.z});
            if (!frustum.sphereVisible(center, gm.bounds.radius * scale)) continue;
            const u64 paletteBytes = static_cast<u64>(std::min(item.boneCount, 64u)) * sizeof(BoneMatrix);
            u64 paletteOffset = 0, instanceOffset = 0;
            u8* palette = m->device.allocUpload(paletteBytes, kRingAlignment, &paletteOffset);
            u8* instance = palette ? m->device.allocUpload(sizeof(InstanceData), kRingAlignment, &instanceOffset) : nullptr;
            if (!instance) {
                m->warnOverflow(paletteBytes);
                continue;
            }
            std::memcpy(palette, list.bones(item), paletteBytes);
            std::memcpy(instance, &inst, sizeof(InstanceData));
            if (!skinBound) cmd.setPipeline(m->shadowSkin), skinBound = true;
            ForwardPush push{m->device.uploadBufferIndex(), static_cast<u32>(m->shadowFrameOffsets[c]), static_cast<u32>(instanceOffset), 0, 0};
            push.paletteOffset = static_cast<u32>(paletteOffset);
            cmd.pushConstants(&push, sizeof(push));
            cmd.setVertexBuffer(gm.vertexBuffer, kSkinnedVertexStride);
            cmd.setIndexBuffer(gm.indexBuffer);
            cmd.drawIndexed(gm.indexCount, 1);
            m->stats.drawCalls += 1;
        }
        if (skinBound) cmd.setPipeline(m->shadowPipeline);
    }
    cmd.endTarget(m->shadowTarget);
}

void Renderer::Impl::writeFrameConstants(const FrameParams& p, u64* offset, bool withShadows) {
    FrameConstants fc{};
    std::memcpy(fc.viewProj, p.viewProj.m, sizeof(fc.viewProj));
    const Vec3 sun = normalize(p.sunDirection);
    const f32 cameraPos[4] = {p.cameraPos.x, p.cameraPos.y, p.cameraPos.z, 0.0f};
    const f32 sunDir[4] = {sun.x, sun.y, sun.z, 0.0f};
    const f32 sunColor[4] = {p.sunColor.x, p.sunColor.y, p.sunColor.z, 0.0f};
    const f32 skyColor[4] = {p.skyColor.x, p.skyColor.y, p.skyColor.z, 0.0f};
    const f32 groundColor[4] = {p.groundColor.x, p.groundColor.y, p.groundColor.z, 0.0f};
    const f32 fogColor[4] = {p.fogColor.x, p.fogColor.y, p.fogColor.z, p.fogDensity};
    const f32 tint[4] = {p.tintColor.x, p.tintColor.y, p.tintColor.z, std::clamp(p.tintAmount, 0.0f, 1.0f)};
    std::memcpy(fc.cameraPos, cameraPos, sizeof(cameraPos));
    std::memcpy(fc.sunDir, sunDir, sizeof(sunDir));
    std::memcpy(fc.sunColor, sunColor, sizeof(sunColor));
    std::memcpy(fc.skyColor, skyColor, sizeof(skyColor));
    std::memcpy(fc.groundColor, groundColor, sizeof(groundColor));
    std::memcpy(fc.fogColor, fogColor, sizeof(fogColor));
    std::memcpy(fc.screenTint, tint, sizeof(tint));
    Vec3 wind{p.windDirection.x, 0.0f, p.windDirection.z};
    const f32 windLength = std::sqrt(wind.x * wind.x + wind.z * wind.z);
    wind = windLength > 1e-6f ? wind * (1.0f / windLength) : Vec3{1.0f, 0.0f, 0.0f};
    const f32 windValues[4] = {wind.x, wind.z, std::max(p.windStrength, 0.0f), p.time};
    std::memcpy(fc.wind, windValues, sizeof(windValues));
    fc.style[0] = device.textureIndex(toonRamp);
    invert(&p.viewProj.m[0][0], fc.invViewProj);
    if (p.environment && p.environment <= environments.size()) {
        const Environment& e = environments[p.environment - 1];
        const u32 env[4] = {e.sky, e.reflect, e.reflectMips, 1u};
        std::memcpy(fc.env, env, sizeof(env));
        const f32 params[4] = {std::max(p.environmentIntensity, 0.0f), std::cos(p.environmentRotation), std::sin(p.environmentRotation), 0.0f};
        std::memcpy(fc.envParams, params, sizeof(params));
        for (u32 i = 0; i < 9; ++i) std::memcpy(fc.sh[i], e.sh[i], sizeof(e.sh[i]));
    }
    if (p.probes && p.probeCount) {
        struct ProbeGpu {
            f32 position[3];
            u32 cube;
            f32 halfExtents[3];
            u32 pad;
        };
        std::vector<ProbeGpu> table;
        for (u32 i = 0; i < p.probeCount && table.size() < kMaxReflectionProbes; ++i) {
            const ReflectionProbe& probe = p.probes[i];
            if (probe.id && probe.id <= probes.size())
                table.push_back({{probe.position.x, probe.position.y, probe.position.z}, probes[probe.id - 1],
                                 {probe.halfExtents.x, probe.halfExtents.y, probe.halfExtents.z}, 0});
        }
        u64 tableOffset = 0;
        const u64 bytes = table.size() * sizeof(ProbeGpu);
        if (u8* cpu = bytes ? device.allocUpload(bytes, kRingAlignment, &tableOffset) : nullptr) {
            std::memcpy(cpu, table.data(), bytes);
            fc.probes[0] = static_cast<u32>(tableOffset);
            fc.probes[1] = static_cast<u32>(table.size());
        } else if (bytes) warnOverflow(bytes);
    }
    frameLights.assign(p.lights, p.lights + (p.lights ? std::min(p.lightCount, kMaxPointLights) : 0u));
    if (!frameLights.empty()) {
        struct LightGpu {
            f32 position[3];
            f32 range;
            f32 color[3];
            f32 pad;
        };
        std::vector<LightGpu> table;
        table.reserve(frameLights.size());
        for (const PointLight& l : frameLights)
            table.push_back({{l.position.x, l.position.y, l.position.z}, std::max(l.range, 1e-3f), {l.color.x, l.color.y, l.color.z}, 0.0f});
        u64 tableOffset = 0;
        const u64 bytes = table.size() * sizeof(LightGpu);
        if (u8* cpu = device.allocUpload(bytes, kRingAlignment, &tableOffset)) {
            std::memcpy(cpu, table.data(), bytes);
            fc.lights[0] = static_cast<u32>(tableOffset);
            fc.lights[1] = static_cast<u32>(table.size());
        } else {
            warnOverflow(bytes);
            frameLights.clear();
        }
    }
    if (withShadows) computeCascades(p, fc);
    fc.materials[0] = static_cast<u32>(materialOffset);
    fc.materials[1] = static_cast<u32>(materials.size());
    if (u8* cpu = device.allocUpload(sizeof(fc), kRingAlignment, offset)) std::memcpy(cpu, &fc, sizeof(fc));
    else warnOverflow(sizeof(fc));
}

void Renderer::Impl::drawSky(const FrameParams& p) {
    if (!p.environment || p.environment > environments.size()) return;
    rhi::CommandList& cmd = device.cmd();
    cmd.setPipeline(sky);
    const ForwardPush push{device.uploadBufferIndex(), static_cast<u32>(frameOffset), 0, 0, 0};
    cmd.pushConstants(&push, sizeof(push));
    cmd.draw(3, 1);
    stats.drawCalls += 1;
}

u32 Renderer::createEnvironment(const EnvironmentData& d) {
    if (!m || !d.sky || !d.reflect || !d.skySize || !d.reflectSize || !d.reflectMips) return 0;
    Impl::Environment e;
    rhi::TextureInit sky;
    sky.width = sky.height = d.skySize;
    sky.format = rhi::Format::Rgb9e5;
    sky.data = d.sky;
    sky.cube = true;
    rhi::TextureInit reflect;
    reflect.width = reflect.height = d.reflectSize;
    reflect.mipLevels = d.reflectMips;
    reflect.format = rhi::Format::Rgb9e5;
    reflect.data = d.reflect;
    reflect.cube = true;
    e.sky = createTexture(sky);
    e.reflect = createTexture(reflect);
    e.reflectMips = d.reflectMips;
    std::memcpy(e.sh, d.sh, sizeof(e.sh));
    m->environments.push_back(e);
    return static_cast<u32>(m->environments.size());
}

u32 Renderer::createProbe(u32 size, u32 mips, const u32* reflect) {
    if (!m || !reflect || !size || !mips || mips > 255) return 0;
    rhi::TextureInit init;
    init.width = init.height = size;
    init.mipLevels = mips;
    init.format = rhi::Format::Rgb9e5;
    init.data = reflect;
    init.cube = true;
    m->probes.push_back(createTexture(init) | (mips << 24));
    return static_cast<u32>(m->probes.size());
}

u32 Renderer::createLut(const f32* rgb) {
    if (!m || !rgb) return 0;
    Impl::Lut lut;
    lut.rgb.assign(rgb, rgb + static_cast<size_t>(kLutSize) * kLutSize * kLutSize * 3);
    lut.texture = m->device.createTexture(kLutWidth, kLutSize, rhi::Format::Rgba8Unorm);
    m->luts.push_back(std::move(lut));
    m->uploadLut(static_cast<u32>(m->luts.size() - 1));
    return static_cast<u32>(m->luts.size() - 1);
}

void Renderer::Impl::uploadLut(u32 index) {
    const Lut& lut = luts[index];
    TextureUpload u;
    u.texture = lut.texture;
    u.bpp = 4;
    u.x = u.y = 0;
    u.w = kLutWidth;
    u.h = kLutSize;
    u.pixels.resize(static_cast<size_t>(kLutWidth) * kLutSize * 4);
    for (u32 b = 0; b < kLutSize; ++b)
        for (u32 g = 0; g < kLutSize; ++g)
            for (u32 r = 0; r < kLutSize; ++r) {
                const size_t cell = (static_cast<size_t>(b) * kLutSize + g) * kLutSize + r;
                f32 c[3] = {lut.rgb[cell * 3], lut.rgb[cell * 3 + 1], lut.rgb[cell * 3 + 2]};
                daltonize(postOptions.colorBlind, c);
                u8* px = &u.pixels[(static_cast<size_t>(g) * kLutWidth + b * kLutSize + r) * 4];
                for (u32 i = 0; i < 3; ++i) px[i] = static_cast<u8>(std::clamp(c[i], 0.0f, 1.0f) * 255.0f + 0.5f);
                px[3] = 255;
            }
    uploads.push_back(std::move(u));
}

void Renderer::setPostOptions(const PostOptions& options) {
    if (!m) return;
    const bool relut = options.colorBlind != m->postOptions.colorBlind;
    m->postOptions = options;
    m->postOptions.brightness = std::clamp(options.brightness, 0.1f, 4.0f);
    if (relut)
        for (u32 i = 0; i < m->luts.size(); ++i) m->uploadLut(i);
}

const PostOptions& Renderer::postOptions() const {
    static const PostOptions none{};
    return m ? m->postOptions : none;
}

void Renderer::Impl::resizeChain(PostChain& chain, u32 width, u32 height) {
    device.destroyRenderTarget(chain.scene);
    for (rhi::RenderTarget& b : chain.bloom) device.destroyRenderTarget(b), b = {};
    chain.scene = device.createRenderTarget(width, height, rhi::TargetKind::Scene);
    chain.width = width;
    chain.height = height;
}

void Renderer::Impl::ensureBloom(PostChain& chain) {
    if (chain.bloom[0]) return;
    for (u32 i = 0; i < kBloomLevels; ++i) {
        chain.bloomSize[i][0] = std::max(1u, chain.width >> (i + 1));
        chain.bloomSize[i][1] = std::max(1u, chain.height >> (i + 1));
        chain.bloom[i] = device.createRenderTarget(chain.bloomSize[i][0], chain.bloomSize[i][1], rhi::TargetKind::Hdr);
    }
}

void Renderer::Impl::runPost(PostChain& chain, const PostParams& post, const f32 rect[4], rhi::RenderTarget output, bool timed,
                             f32 time) {
    rhi::CommandList& cmd = device.cmd();
    if (timed) beginPass(GpuPass::Post);
    const f32 w = static_cast<f32>(chain.width), h = static_cast<f32>(chain.height);
    const f32 region[4] = {rect[0] / w, rect[1] / h, (rect[0] + rect[2]) / w, (rect[1] + rect[3]) / h};
    const bool tone = post.toneMapping;
    const bool bloom = tone && postOptions.bloom && post.bloomIntensity > 0.0f;
    const u32 lut = post.lut < luts.size() ? post.lut : 0;
    if (bloom) ensureBloom(chain); // 처음 쓸 때 (만들기는 프레임 안에서도 된다: 그림자 맵과 같다)
    const bool bloomReady = bloom && chain.bloom[0];

    PostConstants pc{};
    const f32 exposure[4] = {std::exp2(post.exposure) * postOptions.brightness, std::max(post.bloomIntensity, 0.0f),
                             std::clamp(post.vignette, 0.0f, 1.0f), std::clamp(post.vignetteSmoothness, 0.0f, 1.0f)};
    const f32 effects[4] = {postOptions.filmGrain ? std::clamp(post.grain, 0.0f, 1.0f) : 0.0f,
                            postOptions.chromaticAberration ? std::clamp(post.chromaticAberration, 0.0f, 0.1f) : 0.0f,
                            std::fmod(time, 64.0f), 0.0f};
    const u32 flags = tone ? kPostToneMap | kPostDither | (postOptions.antiAliasing ? kPostFxaa : 0u) : 0u;
    const u32 textures[4] = {bloomReady ? device.textureIndex(device.renderTargetTexture(chain.bloom[0])) : kNoTexture,
                             tone && (lut != 0 || postOptions.colorBlind != ColorBlindMode::Off) ? device.textureIndex(luts[lut].texture)
                                                                                                 : kNoTexture,
                             flags, 0};
    const f32 bloomColor[4] = {post.bloomTint.x, post.bloomTint.y, post.bloomTint.z, std::max(post.bloomThreshold, 0.0f)};
    const f32 sceneSize[4] = {w, h, 1.0f / w, 1.0f / h};
    std::memcpy(pc.exposure, exposure, sizeof(exposure));
    if (tone) std::memcpy(pc.effects, effects, sizeof(effects));
    std::memcpy(pc.textures, textures, sizeof(textures));
    std::memcpy(pc.bloom, bloomColor, sizeof(bloomColor));
    std::memcpy(pc.region, region, sizeof(region));
    std::memcpy(pc.sceneSize, sceneSize, sizeof(sceneSize));
    u64 constOffset = 0;
    u8* cpu = device.allocUpload(sizeof(pc), kRingAlignment, &constOffset);
    if (!cpu) {
        warnOverflow(sizeof(pc));
        if (timed) endPass();
        return;
    }
    std::memcpy(cpu, &pc, sizeof(pc));
    PostPush push;
    push.ring = device.uploadBufferIndex();
    push.constOffset = static_cast<u32>(constOffset);
    std::memcpy(push.rect, region, sizeof(region));
    const u32 sceneTexture = device.textureIndex(device.renderTargetTexture(chain.scene));

    if (bloomReady) { // 듀얼 필터: 반씩 내려가고 (첫 단계는 문턱) 더하기 섞기로 올라온다
        for (u32 i = 0; i < kBloomLevels; ++i) {
            const f32 srcW = i ? static_cast<f32>(chain.bloomSize[i - 1][0]) : w, srcH = i ? static_cast<f32>(chain.bloomSize[i - 1][1]) : h;
            const f32 dstW = static_cast<f32>(chain.bloomSize[i][0]), dstH = static_cast<f32>(chain.bloomSize[i][1]);
            cmd.beginTarget(chain.bloom[i], nullptr, 0.0f);
            cmd.setViewport(0.0f, 0.0f, dstW, dstH);
            cmd.setPipeline(bloomDown);
            push.source = i ? device.textureIndex(device.renderTargetTexture(chain.bloom[i - 1])) : sceneTexture;
            push.first = i == 0;
            push.texel[0] = 1.0f / srcW, push.texel[1] = 1.0f / srcH;
            push.scale[0] = 1.0f / dstW, push.scale[1] = 1.0f / dstH;
            cmd.pushConstants(&push, sizeof(push));
            cmd.draw(3, 1);
            cmd.endTarget(chain.bloom[i]);
        }
        push.first = 0;
        for (u32 i = kBloomLevels - 1; i-- > 0;) {
            const f32 dstW = static_cast<f32>(chain.bloomSize[i][0]), dstH = static_cast<f32>(chain.bloomSize[i][1]);
            cmd.beginTarget(chain.bloom[i], nullptr, 0.0f);
            cmd.setViewport(0.0f, 0.0f, dstW, dstH);
            cmd.setPipeline(bloomUp);
            push.source = device.textureIndex(device.renderTargetTexture(chain.bloom[i + 1]));
            push.texel[0] = 1.0f / static_cast<f32>(chain.bloomSize[i + 1][0]), push.texel[1] = 1.0f / static_cast<f32>(chain.bloomSize[i + 1][1]);
            push.scale[0] = 1.0f / dstW, push.scale[1] = 1.0f / dstH;
            cmd.pushConstants(&push, sizeof(push));
            cmd.draw(3, 1);
            cmd.endTarget(chain.bloom[i]);
        }
        stats.drawCalls += kBloomLevels * 2 - 1;
    }

    if (output) {
        cmd.beginTarget(output, nullptr, 0.0f);
        cmd.setViewport(0.0f, 0.0f, w, h);
    } else {
        cmd.beginOverlay();
        cmd.setViewport(rect[0], rect[1], rect[2], rect[3]);
    }
    cmd.setPipeline(uber);
    push.source = sceneTexture;
    push.texel[0] = 1.0f / w, push.texel[1] = 1.0f / h;
    push.scale[0] = 1.0f / w, push.scale[1] = 1.0f / h;
    cmd.pushConstants(&push, sizeof(push));
    cmd.draw(3, 1);
    stats.drawCalls += 1;
    if (output) cmd.endTarget(output);
    if (timed) endPass();
}

void Renderer::Impl::resolveScene() {
    if (!sceneBound || sceneResolved) return;
    sceneResolved = true;
    device.cmd().endTarget(mainChain.scene);
    PostParams post = sceneParams.post;
    if (debugView != DebugView::None) post.toneMapping = false; // 보기 모드는 색을 그대로
    runPost(mainChain, post, sceneViewport, {}, true, sceneParams.time);
}

rhi::RenderTarget Renderer::createRenderTarget(u32 width, u32 height) {
    if (!m || width == 0 || height == 0) return {};
    Impl::Offscreen o;
    o.output = m->device.createRenderTarget(width, height, rhi::TargetKind::Ldr);
    m->resizeChain(o.chain, width, height); // 짝 HDR 장면 대상
    if (m->offscreen.size() <= o.output.id) m->offscreen.resize(o.output.id + 1);
    m->offscreen[o.output.id] = o;
    return o.output;
}

rhi::Texture Renderer::renderTargetTexture(rhi::RenderTarget target) const {
    return m ? m->device.renderTargetTexture(target) : rhi::Texture{};
}

void Renderer::drawToTarget(rhi::RenderTarget target, const FrameParams& p, const DrawList& list, LightingModel lighting) {
    if (!m || !m->inFrame || !target || target.id >= m->offscreen.size() || m->offscreen[target.id].output.id != target.id) return;
    m->ensureScene(); // 본 화면이 아직 지워지지 않았으면 먼저
    m->resolveScene(); // 본 화면은 여기서 포스트를 거쳐 백버퍼로 (이 뒤로 본 화면 3D는 그리지 않는다)
    Impl::PostChain& chain = m->offscreen[target.id].chain;
    rhi::CommandList& cmd = m->device.cmd();
    // 본 화면의 상태를 잠시 바꾼다: 프레임 상수·프러스텀·보기 모드 (오프스크린은 늘 조명 그림)
    const u64 frameOffset = m->frameOffset;
    const Frustum frustum = m->frustum;
    const DebugView debugView = m->debugView;
    const LightingModel projectLighting = m->lighting;
    std::vector<PointLight> frameLights = m->frameLights;
    m->debugView = DebugView::None;
    m->lighting = lighting;
    m->writeFrameConstants(p, &m->frameOffset, false); // 오프스크린은 그림자 없이 (룩데브, 반사 프로브 굽기)
    m->frustum = Frustum::fromViewProj(p.viewProj);

    const Vec3 sky = p.fogColor;
    const f32 clear[4] = {sky.x, sky.y, sky.z, 1.0f};
    cmd.beginTarget(chain.scene, clear, 0.0f);
    const f32 rect[4] = {0.0f, 0.0f, static_cast<f32>(chain.width), static_cast<f32>(chain.height)};
    cmd.setViewport(rect[0], rect[1], rect[2], rect[3]);
    m->drawSky(p);
    cmd.setPipeline(m->scenePipeline());
    list.submit(*this);
    cmd.endTarget(chain.scene);
    m->runPost(chain, p.post, rect, target, false, p.time);

    m->frameOffset = frameOffset;
    m->frustum = frustum;
    m->debugView = debugView;
    m->lighting = projectLighting;
    m->frameLights = std::move(frameLights);
}

void Renderer::drawInstances(MeshId meshId, const InstanceData* instances, u32 count) {
    if (!m || !m->inFrame || count == 0 || meshId >= m->meshes.size()) return;
    m->ensureScene();
    if (m->meshes[meshId].skinned) { // 팔레트 없이 add한 스킨 메시: 바인드 자세 (단위 팔레트)
        static const std::vector<BoneMatrix> identity(64, BoneMatrix{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}});
        for (u32 i = 0; i < count; ++i) drawSkinned(meshId, instances[i], identity.data(), 64);
        return;
    }
    m->drawMesh(meshId, instances, count, 0);
}

bool Renderer::meshSkinned(MeshId mesh) const { return m && mesh < m->meshes.size() && m->meshes[mesh].skinned; }

void Renderer::drawSkinned(MeshId meshId, const InstanceData& instance, const BoneMatrix* bones, u32 boneCount) {
    if (!m || !m->inFrame || meshId >= m->meshes.size() || !bones || boneCount == 0) return;
    m->ensureScene();
    u64 paletteOffset = 0;
    const u64 bytes = static_cast<u64>(std::min(boneCount, 64u)) * sizeof(BoneMatrix);
    u8* cpu = m->device.allocUpload(bytes, kRingAlignment, &paletteOffset);
    if (!cpu) {
        m->warnOverflow(bytes);
        return;
    }
    std::memcpy(cpu, bones, bytes);
    m->drawMesh(meshId, &instance, 1, static_cast<u32>(paletteOffset));
}

void Renderer::Impl::drawMesh(MeshId meshId, const InstanceData* instances, u32 count, u32 paletteOffset) {
    const u64 bytes = static_cast<u64>(count) * sizeof(InstanceData);
    u64 offset = 0;
    u8* cpu = device.allocUpload(bytes, kRingAlignment, &offset);
    if (!cpu) {
        warnOverflow(bytes);
        return;
    }
    std::memcpy(cpu, instances, bytes);

    const GpuMesh& mesh = meshes[meshId];
    const f32 boundsScale = mesh.skinned ? DrawList::kSkinnedBoundsScale : 1.0f; // 빛 고르기 (애니메이션 여유)
    u64 lightOffset = 0;
    if (!frameLights.empty()) { // 점광원: 인스턴스마다 영향이 큰 빛을 고른다 (바운딩 구와 닿는 빛, 중심에서의 밝기 순)
        lightPicks.assign(static_cast<size_t>(count) * kMaxLightsPerObject, 0xFF);
        struct Pick {
            f32 score;
            u32 light;
        };
        Pick best[kMaxLightsPerObject];
        for (u32 i = 0; i < count; ++i) {
            const InstanceData& inst = instances[i];
            const Vec3 s = inst.scale;
            const f32 radius = mesh.bounds.radius * boundsScale * std::max({std::abs(s.x), std::abs(s.y), std::abs(s.z)});
            const Vec3 center =
                inst.position + rotate(inst.rotation, Vec3{mesh.bounds.center.x * s.x, mesh.bounds.center.y * s.y, mesh.bounds.center.z * s.z});
            u32 picked = 0;
            for (u32 l = 0; l < frameLights.size(); ++l) {
                const PointLight& light = frameLights[l];
                const Vec3 d = light.position - center;
                const f32 dist = std::sqrt(dot(d, d));
                if (dist > light.range + radius) continue;
                const f32 near = std::max(dist - radius, 0.0f); // 물체에서 가장 가까운 점
                const f32 score = (light.color.x + light.color.y + light.color.z) * (1.0f - near / light.range) / (near * near + 1.0f);
                u32 at = picked < lightsPerObject ? picked++ : lightsPerObject;
                if (at == lightsPerObject) { // 가득 찼다: 가장 약한 것보다 세면 바꾼다
                    u32 weakest = 0;
                    for (u32 k = 1; k < picked; ++k)
                        if (best[k].score < best[weakest].score) weakest = k;
                    if (best[weakest].score >= score) continue;
                    at = weakest;
                }
                best[at] = {score, l};
            }
            for (u32 k = 0; k < picked; ++k) lightPicks[static_cast<size_t>(i) * kMaxLightsPerObject + k] = static_cast<u8>(best[k].light);
        }
        if (u8* dst = device.allocUpload(lightPicks.size(), kRingAlignment, &lightOffset)) {
            std::memcpy(dst, lightPicks.data(), lightPicks.size());
        } else {
            warnOverflow(lightPicks.size());
            return;
        }
    }
    rhi::CommandList& cmd = device.cmd();
    ForwardPush push{device.uploadBufferIndex(), static_cast<u32>(frameOffset), static_cast<u32>(offset),
                     static_cast<u32>(debugView), static_cast<u32>(lightOffset)};
    push.paletteOffset = paletteOffset;
    if (mesh.skinned) cmd.setPipeline(scenePipeline(true));
    cmd.pushConstants(&push, sizeof(push));
    cmd.setVertexBuffer(mesh.vertexBuffer, mesh.skinned ? kSkinnedVertexStride : sizeof(Vertex));
    cmd.setIndexBuffer(mesh.indexBuffer);
    cmd.drawIndexed(mesh.indexCount, count);
    stats.drawCalls += 1;
    stats.instances += count;
    if (mesh.outline && debugView == DebugView::None) { // 외곽선: 같은 인스턴스를 껍질 패스로 한 번 더
        cmd.setPipeline(mesh.skinned ? outlineSkin : outline);
        cmd.pushConstants(&push, sizeof(push));
        cmd.drawIndexed(mesh.indexCount, count);
        cmd.setPipeline(scenePipeline());
        stats.drawCalls += 1;
    } else if (mesh.skinned) {
        cmd.setPipeline(scenePipeline());
    }
}

void Renderer::drawParticles(const Particle* particles, u32 count) {
    if (!m || !m->inFrame || !particles || count == 0) return;
    m->ensureScene();
    count = std::min(count, kMaxParticlesPerFrame);
    const u64 bytes = static_cast<u64>(count) * sizeof(Particle);
    u64 offset = 0;
    u8* cpu = m->device.allocUpload(bytes, kRingAlignment, &offset);
    if (!cpu) {
        m->warnOverflow(bytes);
        return;
    }
    std::memcpy(cpu, particles, bytes);
    rhi::CommandList& cmd = m->device.cmd();
    const bool vfxPass = m->openPass == static_cast<i32>(GpuPass::Opaque); // 오프스크린(룩데브)은 따로 재지 않는다
    if (vfxPass) m->beginPass(GpuPass::Vfx);
    cmd.setPipeline(m->particles);
    const ForwardPush push{m->device.uploadBufferIndex(), static_cast<u32>(m->frameOffset), static_cast<u32>(offset), 0, 0};
    cmd.pushConstants(&push, sizeof(push));
    cmd.draw(6, count);
    cmd.setPipeline(m->scenePipeline()); // 뒤에 더 그리는 3D가 있으면 (drawToTarget 등)
    m->stats.drawCalls += 1; // Vfx 구간은 beginOverlay(UI)까지 (파티클은 DrawList::submit의 마지막)
}

void Renderer::beginOverlay() {
    if (!m || !m->inFrame) return;
    m->ensureScene();
    m->resolveScene();
    m->device.cmd().beginOverlay();
    m->beginPass(GpuPass::Ui); // UI + 디버그 UI
}

void Renderer::updateUiAtlas(UiAtlas atlas, u32 x, u32 y, u32 w, u32 h, const u8* pixels) {
    if (!m || w == 0 || h == 0 || x + w > kUiAtlasSize || y + h > kUiAtlasSize) return;
    const u32 bpp = atlas == UiAtlas::Glyphs ? 1u : 4u;
    // 큰 영역은 줄 묶음으로 나눈다 (프레임 업로드 링에 들어가게)
    const u32 rowsPerPiece = std::max(1u, static_cast<u32>((1u << 20) / (w * bpp)));
    for (u32 row = 0; row < h; row += rowsPerPiece) {
        Impl::TextureUpload u;
        u.texture = m->uiAtlas[static_cast<u32>(atlas)];
        u.bpp = bpp;
        u.x = x;
        u.y = y + row;
        u.w = w;
        u.h = std::min(rowsPerPiece, h - row);
        u.pixels.assign(pixels + static_cast<size_t>(row) * w * bpp, pixels + static_cast<size_t>(row + u.h) * w * bpp);
        m->uploads.push_back(std::move(u));
    }
}

void Renderer::drawUi(const UiQuad* quads, u32 count) {
    if (!m || !m->inFrame) return;
    m->ensureScene();
    m->resolveScene();
    m->flushUploads(); // 복사는 그리기 대상을 푼다: 아래에서 오버레이를 다시 묶는다
    if (count == 0) return;
    const u64 bytes = static_cast<u64>(count) * sizeof(UiQuad);
    u64 offset = 0;
    u8* cpu = m->device.allocUpload(bytes, kRingAlignment, &offset);
    if (!cpu) {
        m->warnOverflow(bytes);
        return;
    }
    std::memcpy(cpu, quads, bytes);
    rhi::CommandList& cmd = m->device.cmd();
    cmd.beginOverlay();
    cmd.setViewport(0.0f, 0.0f, static_cast<f32>(width()), static_cast<f32>(height()));
    cmd.setPipeline(m->ui);
    UiPush push;
    push.screenSize[0] = static_cast<f32>(width());
    push.screenSize[1] = static_cast<f32>(height());
    push.ring = m->device.uploadBufferIndex();
    push.quadOffset = static_cast<u32>(offset);
    push.glyphAtlas = m->device.textureIndex(m->uiAtlas[0]);
    push.imageAtlas = m->device.textureIndex(m->uiAtlas[1]);
    cmd.pushConstants(&push, sizeof(push));
    cmd.draw(6, count);
    m->stats.drawCalls += 1;
}

void Renderer::waitIdle() {
    if (m) m->device.waitIdle();
}

rhi::Device& Renderer::device() { return m->device; }

u32 Renderer::width() const { return m ? m->device.width() : 0; }
u32 Renderer::height() const { return m ? m->device.height() : 0; }

void Renderer::endFrame() {
    if (!m || !m->inFrame) return;
    m->ensureScene(); // 아무것도 그리지 않은 프레임도 지운다
    m->resolveScene();
    m->endPass();
    if (m->captureOut) {
        rhi::Capture capture;
        capture.source = m->captureSource;
        m->device.endFrame(m->vsync, &capture);
        m->captureOut->width = capture.width;
        m->captureOut->height = capture.height;
        m->captureOut->rgba = std::move(capture.rgba);
        m->captureOut = nullptr;
    } else if (m->capturePath.empty()) {
        m->device.endFrame(m->vsync);
    } else {
        rhi::Capture capture;
        m->device.endFrame(m->vsync, &capture);
        Image image;
        image.width = capture.width;
        image.height = capture.height;
        image.rgba = std::move(capture.rgba);
        if (image.width && savePng(m->capturePath, image))
            MEALKIT_INFO("Screenshot saved: %s (%ux%u)", m->capturePath.c_str(), image.width, image.height);
        else MEALKIT_WARN("Screenshot failed: %s", m->capturePath.c_str());
        m->capturePath.clear();
    }
    m->inFrame = false;
}

void Renderer::setVsync(bool enabled) {
    if (m) m->vsync = enabled;
}
bool Renderer::vsync() const { return m && m->vsync; }
void Renderer::requestCapture(const char* pngPath) {
    if (m) m->capturePath = pngPath;
}
void Renderer::captureTarget(rhi::RenderTarget target, Image* out) {
    if (!m || !target || !out) return;
    m->captureSource = target;
    m->captureOut = out;
}
const char* Renderer::adapterName() const { return m ? m->device.gpu().adapter.c_str() : ""; }
bool Renderer::isUma() const { return m && m->device.gpu().uma; }
const GpuInfo& Renderer::gpu() const {
    static const GpuInfo empty{};
    return m ? m->device.gpu() : empty;
}
const char* Renderer::initError() const { return m ? m->device.initError().c_str() : ""; }
const RenderStats& Renderer::stats() const {
    static const RenderStats empty{};
    return m ? m->stats : empty;
}
BoundingSphere Renderer::meshBounds(MeshId mesh) const {
    return m && mesh < m->meshes.size() ? m->meshes[mesh].bounds : BoundingSphere{};
}
const Frustum& Renderer::frustum() const {
    static const Frustum none; // 평면이 없으면 모두 보인다
    return m ? m->frustum : none;
}
void Renderer::addCulled(u32 count) {
    if (m) m->stats.culled += count;
}
bool Renderer::supportsDebugView(DebugView view) const {
    if (view == DebugView::None) return true;
    if (!m || view >= DebugView::Count) return false;
#if MEALKIT_DEBUG_UI
    return true;
#else
    return false; // 출시 빌드: 보기 모드 파이프라인이 없다
#endif
}
rhi::GpuMemory Renderer::gpuMemory() const { return m ? m->device.gpuMemory() : rhi::GpuMemory{}; }

const char* gpuPassName(GpuPass pass) {
    switch (pass) {
    case GpuPass::Opaque: return "opaque";
    case GpuPass::Shadow: return "shadow";
    case GpuPass::Post: return "post";
    case GpuPass::Vfx: return "vfx";
    case GpuPass::Ui: return "ui";
    default: return "?";
    }
}

} // namespace mealkit::render
