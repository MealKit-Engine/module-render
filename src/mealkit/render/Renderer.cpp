#include "mealkit/render/Renderer.h"

#include "mealkit/core/Log.h"
#include "mealkit/jobs/Jobs.h"
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
#include "g_SpritePS.h"
#include "g_SpriteVS.h"
#include "g_BloomDownPS.h"
#include "g_BloomUpPS.h"
#include "g_PostVS.h"
#include "g_UberPS.h"
#include "g_UiPS.h"
#include "g_UiVS.h"

namespace mealkit::render {
namespace {

// Same layout as FrameConstants in shaders/Forward.hlsl (written to the upload ring and read by the shader with Load)
struct FrameConstants {
    f32 viewProj[16];
    f32 cameraPos[4];
    f32 sunDir[4];
    f32 sunColor[4];
    f32 skyColor[4];
    f32 groundColor[4];
    f32 fogColor[4];
    f32 screenTint[4];
    u32 materials[4]; // material table ring offset, count, 0, 0
    f32 wind[4];      // horizontal direction x, z, strength, view time
    u32 style[4];     // toon ramp texture (bindless), 0, 0, 0
    f32 invViewProj[16]; // skybox: screen → world direction
    u32 env[4];          // sky cube, reflection cube (g_cubes index), reflection mip count, present (1)
    f32 envParams[4];    // intensity, cos(rotation), sin(rotation), 0
    f32 sh[9][4];        // ambient SH (rgb, 0)
    u32 probes[4];       // reflection probe table ring offset, count, 0, 0 (table: per probe position xyz, cube index | mip count << 24, box half size xyz, 0)
    u32 lights[4];       // point light table ring offset, count, 0, 0 (table: per light position xyz, range, color rgb, 0)
    f32 shadowViewProj[kMaxShadowCascades][16]; // per cascade world → shadow clip (row-major)
    f32 shadowForward[4]; // camera forward direction xyz (cascade selection: forward distance), 0
    f32 shadowSplits[4];  // cascade 0 end distance, last cascade end distance, world size of cascade 0/1 texels (normal offset)
    u32 shadow[4];        // shadow map texture (bindless), cascade count (0 = no shadows), size of one cascade, 0
};
static_assert(sizeof(FrameConstants) == 672);

// 4×4 inverse (row-major, row vector convention kept). Identity for singular matrices
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

// shaders/Forward.hlsl Material (128 bytes)
struct MaterialGpu {
    f32 baseColor[4];
    f32 emissive[3];
    f32 alphaCutoff;
    f32 metallic, roughness, normalScale;
    u32 features;    // kMaterial* bits
    u32 textures[4]; // baseColor, normal, metallicRoughness, emissive (bindless indices, kNoTexture = none)
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

// Pixel shader variants (render::LightingModel order, mealkit_shader LIGHTING_MODEL=n in CMakeLists)
const rhi::ShaderCode kLightingPS[kLightingModelCount] = {
    {g_ForwardPbrPS, sizeof(g_ForwardPbrPS)}, {g_ForwardToonPS, sizeof(g_ForwardToonPS)}, {g_ForwardUnlitPS, sizeof(g_ForwardUnlitPS)}};
const char* const kLightingDefines[kLightingModelCount] = {"LIGHTING_MODEL=0", "LIGHTING_MODEL=1", "LIGHTING_MODEL=2"};

// Toon ramp texture width. Project images are resampled (down or up) to this width with nearest pixels: step edges don't blur
constexpr u32 kToonRampWidth = 256;

struct ForwardPush {
    u32 ring = 0;
    u32 frameOffset = 0;
    u32 instanceOffset = 0;
    u32 debugView = 0;   // render::DebugView (Forward.hlsl PSMain)
    u32 lightOffset = 0; // ring offset of 8 point light indices per instance (8 bytes, 0xFF = end) (read only when the frame has lights)
    u32 paletteOffset = 0; // skinned meshes: ring offset of the palette (48 bytes per joint)
    u32 pad[2] = {};
    f32 boxCenter[4] = {}; // mesh draws: quantization box (GpuVertex position = center + snorm * extent)
    f32 boxExtent[4] = {};
};

// GPU vertex (24 bytes, 32 with SkinWeights): Vertex quantized at upload. The CPU Vertex (36 bytes) stays as is
// (asset caches, physics hull points), so only the GPU copy loses precision
struct GpuVertex {
    i16 position[4]; // snorm in the mesh box (w = 0)
    i16 normal[2];   // octahedral snorm
    u32 color;
    u32 uv;
    u16 material;
    u16 reserved;
};
static_assert(sizeof(GpuVertex) == 24);
constexpr u32 kGpuSkinnedVertexStride = sizeof(GpuVertex) + sizeof(SkinWeights);

i16 toSnorm16(f32 v) { return static_cast<i16>(std::lround(std::clamp(v, -1.0f, 1.0f) * 32767.0f)); }

GpuVertex quantize(const Vertex& v, Vec3 center, Vec3 invExtent) {
    GpuVertex g{};
    const Vec3 p = v.position - center;
    g.position[0] = toSnorm16(p.x * invExtent.x);
    g.position[1] = toSnorm16(p.y * invExtent.y);
    g.position[2] = toSnorm16(p.z * invExtent.z);
    // Octahedral normal: project onto the octahedron, fold the lower half over (decoded in Forward.hlsl decodeOctahedral)
    Vec3 n = v.normal;
    const f32 l1 = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    n = l1 > 0.0f ? n * (1.0f / l1) : Vec3{0.0f, 1.0f, 0.0f};
    f32 ox = n.x, oy = n.y;
    if (n.z < 0.0f) {
        ox = (1.0f - std::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
        oy = (1.0f - std::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);
    }
    g.normal[0] = toSnorm16(ox);
    g.normal[1] = toSnorm16(oy);
    g.color = v.color;
    g.uv = v.uv;
    g.material = v.material;
    return g;
}

struct UiPush {
    f32 screenSize[2] = {};
    u32 ring = 0;
    u32 quadOffset = 0;
    u32 glyphAtlas = 0;
    u32 imageAtlas = 0;
    u32 pad[2] = {};
};

// shaders/Post.hlsl PostPush (48 bytes)
struct PostPush {
    u32 ring = 0;
    u32 constOffset = 0;
    u32 source = 0;
    u32 first = 0;
    f32 rect[4] = {};
    f32 texel[2] = {};
    f32 scale[2] = {};
};

// shaders/Post.hlsl PostConstants (upload ring, 96 bytes)
struct PostConstants {
    f32 exposure[4]; // exposure scale, bloom strength, vignette strength, vignette softness
    f32 effects[4];  // grain, chromatic aberration, time, sharpening
    u32 textures[4]; // top of the bloom chain, LUT, flags, 0 (none = kNoTexture)
    f32 bloom[4];    // bloom color, threshold
    f32 region[4];   // scene region uv
    f32 sceneSize[4];
};
static_assert(sizeof(PostConstants) == 96);
constexpr u32 kPostToneMap = 1u, kPostFxaa = 2u, kPostDither = 4u, kPostSharpen = 8u, kPostPunchy = 16u; // Post.hlsl kPost*
constexpr u32 kLutWidth = kLutSize * kLutSize; // 32³ as a horizontal strip 1024 × 32 (32 cells per blue slice)

// Color blindness correction (Fidaner-style daltonize): simulates the colors that person sees from linear RGB with Machado 2009 (severity 1.0),
// and moves the differences they can't see into channels they can. Input and output are 0..1 sRGB values (LUT cells)
void daltonize(ColorBlindMode mode, f32 c[3]) {
    static const f32 kSimulate[3][9] = {
        {0.152286f, 1.052583f, -0.204868f, 0.114503f, 0.786281f, 0.099216f, -0.003882f, -0.048116f, 1.051998f},  // protanopia (red)
        {0.367322f, 0.860646f, -0.227968f, 0.280085f, 0.672501f, 0.047413f, -0.011820f, 0.042940f, 0.968881f},   // deuteranopia (green)
        {1.255528f, -0.076749f, -0.178779f, -0.078411f, 0.930809f, 0.147602f, 0.004733f, 0.691367f, 0.303900f}}; // tritanopia (blue)
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
    bool outline = false; // at least one outline material: drawn once more with the outline pass
    bool skinned = false; // 32-byte vertices (GpuVertex + SkinWeights), drawn with the skin pipeline
    bool alphaClip = false; // an alpha mask or dissolve material (discard turns off early depth: DrawList draws it last)
    Vec3 boxCenter, boxExtent; // GpuVertex position quantization box (ForwardPush)
    u32 stride() const { return skinned ? kGpuSkinnedVertexStride : sizeof(GpuVertex); }
    void setBox(ForwardPush& push) const {
        push.boxCenter[0] = boxCenter.x, push.boxCenter[1] = boxCenter.y, push.boxCenter[2] = boxCenter.z;
        push.boxExtent[0] = boxExtent.x, push.boxExtent[1] = boxExtent.y, push.boxExtent[2] = boxExtent.z;
    }
};

// Skin pipeline: the same desc with the vertex shader (SKINNED=1) and two joint attributes, 32-byte vertex stride
rhi::PipelineDesc skinnedDesc(rhi::PipelineDesc d, rhi::ShaderCode vs) {
    d.vs = vs;
    d.vertexAttributes.push_back({"JOINTS", rhi::VertexFormat::Uint8x4, sizeof(GpuVertex)});
    d.vertexAttributes.push_back({"WEIGHTS", rhi::VertexFormat::Unorm8x4, sizeof(GpuVertex) + 4});
    d.vertexStride = kGpuSkinnedVertexStride;
    return d;
}

} // namespace

struct Renderer::Impl {
    rhi::Device device;
    CullScratch shadowCull; // drawShadows' per-cascade culling buffers (reused between frames)
    rhi::Pipeline forward[kLightingModelCount]; // one per lighting model (init builds all: switching doesn't hitch)
    rhi::Pipeline outline;                      // outline pass (inverted hull)
    // Skinned mesh variants (only the vertex shader differs: SKINNED=1). The two view modes only in debug UI builds
    rhi::Pipeline forwardSkin[kLightingModelCount], outlineSkin, shadowSkin, wireSkin, overdrawSkin;
    rhi::PipelineDesc forwardSkinDesc[kLightingModelCount], outlineSkinDesc, shadowSkinDesc;
    rhi::Pipeline forwardWire, forwardOverdraw; // editor view modes (debug UI builds only, PBR shader)
    rhi::Pipeline ui;
    rhi::Pipeline particles; // shaders/Particle.hlsl (after opaque 3D, no depth writes)
    rhi::PipelineDesc particleDesc;
    rhi::Pipeline sprites; // shaders/Sprite.hlsl (after opaque 3D, before particles, no depth writes)
    rhi::PipelineDesc spriteDesc;
    rhi::Pipeline sky; // skybox (fullscreen triangle, before 3D without depth)
    rhi::PipelineDesc skyDesc;
    struct Environment {
        u32 sky = 0, reflect = 0, reflectMips = 0; // g_cubes indices
        f32 sh[9][3] = {};
    };
    std::vector<Environment> environments; // createEnvironment index - 1
    std::vector<u32> probes;               // createProbe index - 1 → cube index | mip count << 24
    std::vector<PointLight> frameLights;   // point lights written to this frame's constants (drawInstances picks per instance)
    u32 lightsPerObject = 4;
    std::vector<u8> lightPicks;            // drawInstances work buffer

    // Sun shadows: one depth atlas with the cascades side by side (cascade i at x = i × size)
    ShadowSettings shadowSettings;
    rhi::RenderTarget shadowTarget;
    u32 shadowTargetSize = 0, shadowTargetCascades = 0;
    rhi::Pipeline shadowPipeline;
    rhi::PipelineDesc shadowDesc;
    u32 shadowCascades = 0;                    // this frame (0 = not drawn)
    Mat4 shadowViewProj[kMaxShadowCascades];
    u64 shadowFrameOffsets[kMaxShadowCascades] = {}; // per cascade, frame constants with only viewProj changed
    bool shadowPending = false;                // this frame's shadow map hasn't been drawn yet
    void computeCascades(const FrameParams& p, FrameConstants& fc);

    // Post-processing (shaders/Post.hlsl): HDR scene target + bloom chain (halving from half res, created on first use).
    // The main view is window size (beginFrame matches it); offscreen has a pair of the same size per output target (Ldr)
    static constexpr u32 kBloomLevels = 5;
    struct PostChain {
        rhi::RenderTarget scene;
        rhi::RenderTarget bloom[kBloomLevels];
        u32 width = 0, height = 0;
        u32 bloomSize[kBloomLevels][2] = {};
    };
    PostChain mainChain;
    struct Offscreen {
        rhi::RenderTarget output; // Ldr target given by createRenderTarget (0 = this index is another target)
        PostChain chain;
    };
    std::vector<Offscreen> offscreen; // by RenderTarget::id
    rhi::Pipeline bloomDown, bloomUp, uber;
    rhi::PipelineDesc bloomDownDesc, bloomUpDesc, uberDesc;
    PostOptions postOptions;
    struct Lut {
        std::vector<f32> rgb; // kLutSize³ × 3 (recombined when color blindness correction changes)
        rhi::Texture texture; // kLutWidth × kLutSize RGBA8
    };
    std::vector<Lut> luts; // 0 = identity (for color blindness correction only)
    void uploadLut(u32 index);
    void resizeChain(PostChain& chain, u32 width, u32 height); // outside a frame (old targets are deleted)
    void ensureBloom(PostChain& chain);
    // Scene target (chain.scene, finished through endTarget) to output (or the rect area of the back buffer if none). timed: measured as GPU pass Post
    // scale: the scene is in the rect × scale area (dynamic resolution; the uber pass stretches it to rect)
    void runPost(PostChain& chain, const PostParams& post, const f32 rect[4], rhi::RenderTarget output, bool timed, f32 time,
                 f32 scale = 1.0f);
    // Dynamic resolution (main view only): beginFrame picks the scale from GPU time
    ResolutionSettings resolution;
    f32 renderScale = 1.0f;  // scale chosen by the controller
    f32 sceneScale = 1.0f;   // this frame (1 for view modes)
    f32 gpuAverage = 0.0f;   // exponential average of GPU frame time
    u32 scaleCooldown = 0;   // frames to wait after changing the scale (GPU time lags by kFramesInFlight)
    void updateResolution();
    bool sceneResolved = false; // the main view was sent to the back buffer via post (no main view 3D after this)
    void resolveScene();        // after ensureScene

    // The main view 3D target is bound on first draw (so the shadow pass can draw to another target first)
    bool sceneBound = false;
    f32 sceneClear[4] = {};
    f32 sceneViewport[4] = {};
    bool sceneSky = false;
    FrameParams sceneParams; // environment seen by drawSky (pointer fields aren't used)
    void ensureScene();
    rhi::RenderTarget captureSource;       // captureTarget (this endFrame)
    Image* captureOut = nullptr;
    void drawSky(const FrameParams& p); // after writing frame constants, before 3D (sky HDRI, or the fog → sky color gradient)
    LightingModel lighting = LightingModel::Pbr;
    DebugView debugView = DebugView::None;
    // Shader hot reload swaps only the bytecode and rebuilds. The VS of every forwardDesc[n] is the same
    rhi::PipelineDesc forwardDesc[kLightingModelCount], outlineDesc, uiDesc;
    ShaderReloader shaderReload;
    std::vector<GpuMesh> meshes;
    std::vector<MaterialGpu> materials; // 0 = default white. Written to the upload ring every frame
    std::vector<rhi::Texture> textures;
    rhi::Texture toonRamp; // kToonRampWidth × 1, contents changed by setToonRamp

    // 2D UI (shaders/Ui.hlsl): glyph SDF atlas + image atlas
    rhi::Texture uiAtlas[2];
    struct TextureUpload {
        rhi::Texture texture;
        u32 bpp;
        u32 x, y, w, h;
        std::vector<u8> pixels;
    };
    std::vector<TextureUpload> uploads; // regions not yet uploaded to the GPU (UI atlases, toon ramp. In order)

    u64 frameOffset = 0; // ring offset of this frame's FrameConstants
    u64 materialOffset = 0; // ring offset of this frame's material table (shared by main view and offscreen)
    // Writes to the upload ring and puts the position in offset. withShadows: main view (decides and fills sun shadow cascades)
    void writeFrameConstants(const FrameParams& p, u64* offset, bool withShadows);
    bool vsync = true;
    bool inFrame = false;
    bool uploadOverflowWarned = false;
    RenderStats stats;
    Frustum frustum;
    Vec3 cameraPos; // this frame's (beginFrame)
    Vec3 cameraForward{0.0f, 0.0f, 1.0f};
    i32 openPass = -1; // pass whose GPU time is being measured now (GpuPass, -1 if none)
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

    // View mode pipelines (wire/overdraw): built from the PBR shader (forwardDesc[0]). Skin variants use the skin vertex shader
    void recreateViewModePipelines() {
#if MEALKIT_DEBUG_UI
        device.recreatePipeline(forwardWire, wireDesc());
        device.recreatePipeline(forwardOverdraw, overdrawDesc());
        device.recreatePipeline(wireSkin, skinnedDesc(wireDesc(), forwardSkinDesc[0].vs));
        device.recreatePipeline(overdrawSkin, skinnedDesc(overdrawDesc(), forwardSkinDesc[0].vs));
#endif
    }

    // Shader hot reload (dev builds: recompiles and swaps when engine/shaders changes; does nothing in shipping builds and on PCs
    // without the engine folder). One line per pipeline: put the new bytecode into the desc (the ShaderReloader holds the bytecode)
    // and rebuild. When a PBR variant (normal/skinned) changes, the view mode pipelines derived from it are rebuilt too
    void addShaderReloads() {
        // Two defines on one line: "-D" between them
        static const char* const kSkinDefines[kLightingModelCount] = {"LIGHTING_MODEL=0 -D SKINNED=1", "LIGHTING_MODEL=1 -D SKINNED=1",
                                                                      "LIGHTING_MODEL=2 -D SKINNED=1"};
        struct Reload {
            const char* file;
            rhi::Pipeline* pipeline;
            rhi::PipelineDesc* desc;
            const char* vs;
            const char* ps;
            const char* defines;
        };
        std::vector<Reload> reloads;
        for (u32 i = 0; i < kLightingModelCount; ++i) {
            reloads.push_back({"Forward.hlsl", &forward[i], &forwardDesc[i], "VSMain", "PSMain", kLightingDefines[i]});
            reloads.push_back({"Forward.hlsl", &forwardSkin[i], &forwardSkinDesc[i], "VSMain", "PSMain", kSkinDefines[i]});
        }
        const Reload others[] = {
            {"Forward.hlsl", &outline, &outlineDesc, "VSOutline", "PSOutline", ""},
            {"Forward.hlsl", &outlineSkin, &outlineSkinDesc, "VSOutline", "PSOutline", "SKINNED=1"},
            {"Forward.hlsl", &shadowPipeline, &shadowDesc, "VSShadow", "PSShadow", ""},
            {"Forward.hlsl", &shadowSkin, &shadowSkinDesc, "VSShadow", "PSShadow", "SKINNED=1"},
            {"Forward.hlsl", &sky, &skyDesc, "VSSky", "PSSky", ""},
            {"Particle.hlsl", &particles, &particleDesc, "VSMain", "PSMain", ""},
            {"Sprite.hlsl", &sprites, &spriteDesc, "VSMain", "PSMain", ""},
            {"Ui.hlsl", &ui, &uiDesc, "VSMain", "PSMain", ""},
            {"Post.hlsl", &bloomDown, &bloomDownDesc, "VSMain", "PSBloomDown", ""},
            {"Post.hlsl", &bloomUp, &bloomUpDesc, "VSMain", "PSBloomUp", ""},
            {"Post.hlsl", &uber, &uberDesc, "VSMain", "PSUber", ""},
        };
        reloads.insert(reloads.end(), std::begin(others), std::end(others));
        for (const Reload& r : reloads) {
            const bool viewModeSource = r.desc == &forwardDesc[0] || r.desc == &forwardSkinDesc[0];
            shaderReload.add(
                r.file,
                [this, r, viewModeSource](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
                    r.desc->vs = vs;
                    r.desc->ps = ps;
                    const bool ok = device.recreatePipeline(*r.pipeline, *r.desc);
                    if (ok && viewModeSource) recreateViewModePipelines();
                    return ok;
                },
                r.vs, r.ps, r.defines);
        }
    }

    // Copies pending atlas regions into this frame's upload ring (next frame if there's no room)
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
                                                  : forward[0]; // view modes use colors independent of lighting (inside the PBR shader)
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
    jobs::acquire(); // culling workers (shared with physics); released in shutdown
    m->vsync = desc.vsync;
    if (!m->device.init({desc.window, desc.width, desc.height, desc.debugLayer, desc.gpuCapture, desc.pipelineCache})) return false;

    // Style presets: all three lighting models + the outline pipeline are built here (once during loading, so no unseen combination appears during play)
    rhi::PipelineDesc forward;
    forward.vs = {g_ForwardVS, sizeof(g_ForwardVS)};
    forward.vertexAttributes = {{"POSITION", rhi::VertexFormat::Snorm16x4, 0},
                                {"NORMAL", rhi::VertexFormat::Snorm16x2, 8},
                                {"COLOR", rhi::VertexFormat::Unorm8x4, 12},
                                {"TEXCOORD", rhi::VertexFormat::Half2, 16},
                                {"MATERIAL", rhi::VertexFormat::Uint16x2, 20}};
    forward.vertexStride = sizeof(GpuVertex);
    forward.target = rhi::Target::Scene;
    for (u32 i = 0; i < kLightingModelCount; ++i) {
        m->forwardDesc[i] = forward;
        m->forwardDesc[i].ps = kLightingPS[i];
        m->forward[i] = m->device.createPipeline(m->forwardDesc[i]);
    }
    m->outlineDesc = forward;
    m->outlineDesc.vs = {g_OutlineVS, sizeof(g_OutlineVS)};
    m->outlineDesc.ps = {g_OutlinePS, sizeof(g_OutlinePS)};
    m->outlineDesc.cullFront = true; // only the back faces of the inflated hull: doesn't cover the object in front, remains as a rim around it
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
    ui.premultipliedBlend = true; // premultiplied alpha
    m->ui = m->device.createPipeline(ui);
    m->uiDesc = ui;
    rhi::PipelineDesc sky;
    sky.vs = {g_SkyVS, sizeof(g_SkyVS)};
    sky.ps = {g_SkyPS, sizeof(g_SkyPS)};
    sky.target = rhi::Target::Scene;
    sky.cullBack = false;
    sky.depthTest = false; // drawn first and covered by 3D (no depth writes either)
    m->skyDesc = sky;
    m->sky = m->device.createPipeline(sky);
    rhi::PipelineDesc shadow = forward; // shadows: same vertex format, depth only (only alpha mask/dissolve discard in the pixel shader)
    shadow.vs = {g_ShadowVS, sizeof(g_ShadowVS)};
    shadow.ps = {g_ShadowPS, sizeof(g_ShadowPS)};
    shadow.target = rhi::Target::Shadow;
    shadow.cullBack = false; // double-sided: thin objects and open models cast shadows without gaps (normal offset prevents light leaks)
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
    rhi::PipelineDesc particles;
    particles.vs = {g_ParticleVS, sizeof(g_ParticleVS)};
    particles.ps = {g_ParticlePS, sizeof(g_ParticlePS)};
    particles.target = rhi::Target::Scene;
    particles.cullBack = false;
    particles.premultipliedBlend = true; // additive particles output alpha 0
    particles.depthWrite = false;
    m->particleDesc = particles;
    m->particles = m->device.createPipeline(particles);
    rhi::PipelineDesc sprites = particles; // same state: scene target, both faces, premultiplied blend, depth test without writes
    sprites.vs = {g_SpriteVS, sizeof(g_SpriteVS)};
    sprites.ps = {g_SpritePS, sizeof(g_SpritePS)};
    m->spriteDesc = sprites;
    m->sprites = m->device.createPipeline(sprites);
    // Post: fullscreen triangle (no vertex buffer/depth). The bloom chain is HDR, the uber pass targets the back buffer/Ldr targets
    rhi::PipelineDesc post;
    post.vs = {g_PostVS, sizeof(g_PostVS)};
    post.cullBack = false;
    post.depthTest = false;
    post.target = rhi::Target::Hdr;
    m->bloomDownDesc = post;
    m->bloomDownDesc.ps = {g_BloomDownPS, sizeof(g_BloomDownPS)};
    m->bloomUpDesc = post;
    m->bloomUpDesc.ps = {g_BloomUpPS, sizeof(g_BloomUpPS)};
    m->bloomUpDesc.additiveBlend = true; // stacks on top of the value that came down at that step
    m->uberDesc = post;
    m->uberDesc.ps = {g_UberPS, sizeof(g_UberPS)};
    m->uberDesc.target = rhi::Target::Overlay;
    m->bloomDown = m->device.createPipeline(m->bloomDownDesc);
    m->bloomUp = m->device.createPipeline(m->bloomUpDesc);
    m->uber = m->device.createPipeline(m->uberDesc);
    m->addShaderReloads();

    createMaterial({}); // index 0: default white (meshes without material, boxes)
    std::vector<f32> identity(static_cast<size_t>(kLutSize) * kLutSize * kLutSize * 3);
    for (u32 i = 0; i < kLutSize * kLutSize * kLutSize; ++i) {
        identity[i * 3 + 0] = static_cast<f32>(i % kLutSize) / (kLutSize - 1);
        identity[i * 3 + 1] = static_cast<f32>(i / kLutSize % kLutSize) / (kLutSize - 1);
        identity[i * 3 + 2] = static_cast<f32>(i / (kLutSize * kLutSize)) / (kLutSize - 1);
    }
    createLut(identity.data()); // index 0: identity (for color blindness correction only)
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
            const u32 sx = std::min(ramp->width - 1, x * ramp->width / kToonRampWidth); // first row, nearest pixel
            std::memcpy(px, &ramp->rgba[sx * 4], 4);
        } else {
            // Engine default 3 steps (sRGB; the shader squares them to linear 0.35 / 0.72 / 1.0): shade → mid → lit
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
    jobs::release();
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
    if (m->materials.size() >= 0xFFFF) { // vertex material is 16-bit
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

void Renderer::replaceMesh(MeshId id, const MeshData& data) {
    if (!m || id >= m->meshes.size() || data.vertices.empty() || data.indices.empty()) return;
    const MeshId fresh = createMesh(data);
    GpuMesh& old = m->meshes[id];
    m->device.destroyBuffer(old.vertexBuffer);
    m->device.destroyBuffer(old.indexBuffer);
    old = m->meshes[fresh];
    m->meshes.pop_back();
}

MeshId Renderer::createMesh(const MeshData& data) {
    if (data.vertices.empty() || data.indices.empty()) {
        MEALKIT_WARN("createMesh: empty mesh");
        return 0;
    }
    GpuMesh mesh;
    mesh.skinned = data.skin.size() == data.vertices.size();
    // Quantization box: the vertices' axis-aligned bounds (a flat axis gets extent 1 so it doesn't divide by zero)
    Vec3 lo = data.vertices[0].position, hi = lo;
    for (const Vertex& v : data.vertices) {
        const Vec3 p = v.position;
        lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    mesh.boxCenter = (lo + hi) * 0.5f;
    mesh.boxExtent = (hi - lo) * 0.5f;
    for (f32* e : {&mesh.boxExtent.x, &mesh.boxExtent.y, &mesh.boxExtent.z})
        if (*e <= 0.0f) *e = 1.0f;
    const Vec3 invExtent{1.0f / mesh.boxExtent.x, 1.0f / mesh.boxExtent.y, 1.0f / mesh.boxExtent.z};
    // GpuVertex, with SkinWeights appended for skinned meshes
    const u32 stride = mesh.stride();
    std::vector<u8> packed(data.vertices.size() * stride);
    for (size_t i = 0; i < data.vertices.size(); ++i) {
        const GpuVertex g = quantize(data.vertices[i], mesh.boxCenter, invExtent);
        std::memcpy(&packed[i * stride], &g, sizeof(GpuVertex));
        if (mesh.skinned) std::memcpy(&packed[i * stride + sizeof(GpuVertex)], &data.skin[i], sizeof(SkinWeights));
    }
    mesh.vertexBuffer = m->device.createBuffer(rhi::BufferUsage::Vertex, packed.data(), packed.size());
    // With few vertices, shrink to 16-bit indices to save memory bandwidth (important on integrated graphics)
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
    for (const Vertex& v : data.vertices) {
        if (v.material >= m->materials.size()) continue;
        const MaterialGpu& mat = m->materials[v.material];
        mesh.outline |= (mat.features & kMaterialOutline) != 0;
        mesh.alphaClip |= mat.alphaCutoff > 0.0f || (mat.features & kMaterialDissolve) != 0;
    }
    m->meshes.push_back(mesh);
    return static_cast<MeshId>(m->meshes.size() - 1);
}

bool Renderer::beginFrame(const FrameParams& p) {
    if (m) m->shaderReload.update(); // outside a frame (waits for the GPU when swapping pipelines)
    if (m && width() && height() && (m->mainChain.width != width() || m->mainChain.height != height()))
        m->resizeChain(m->mainChain, width(), height()); // the HDR scene target is window size (deletion happens outside a frame)
    if (!m || !m->mainChain.scene || !m->device.beginFrame()) return false;
    rhi::CommandList& cmd = m->device.cmd();

    const Vec3 sky = lerp(p.fogColor, p.tintColor, std::clamp(p.tintAmount, 0.0f, 1.0f)); // the background gets the flash too
    m->debugView = supportsDebugView(p.debugView) ? p.debugView : DebugView::None;
    const bool overdraw = m->debugView == DebugView::Overdraw;
    const f32 clear[4] = {overdraw ? 0.0f : sky.x, overdraw ? 0.0f : sky.y, overdraw ? 0.0f : sky.z, 1.0f};
    m->flushUploads(); // toon ramp/UI atlases (copies happen before binding draw targets)
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
    m->updateResolution();
    m->sceneScale = m->debugView == DebugView::None ? m->renderScale : 1.0f;

    m->materialOffset = 0;
    const u64 materialBytes = m->materials.size() * sizeof(MaterialGpu);
    if (u8* mat = m->device.allocUpload(materialBytes, kRingAlignment, &m->materialOffset)) std::memcpy(mat, m->materials.data(), materialBytes);
    else m->warnOverflow(materialBytes);
    m->shadowCascades = 0;
    m->writeFrameConstants(p, &m->frameOffset, true); // also decides shadow cascades (shadowCascades)
    if (m->shadowCascades) { // per cascade, frame constants with only viewProj changed (vertex shader of the shadow pass)
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
    m->cameraPos = p.cameraPos;
    m->cameraForward = p.cameraForward;
    const rhi::GpuTimes& gpu = m->device.gpuTimes();
    m->stats.gpuValid = gpu.valid;
    m->stats.gpuFrameMs = gpu.frameMs;
    for (u32 i = 0; i < kGpuPassCount; ++i) m->stats.gpuPassMs[i] = gpu.timerMs[i];
    m->stats.resolutionScale = m->sceneScale;
    m->inFrame = true;
    return true;
}

void Renderer::Impl::ensureScene() {
    if (sceneBound) return;
    sceneBound = true;
    shadowPending = false; // no shadow map drawing after binding the main view (this frame's is empty or already drawn)
    rhi::CommandList& cmd = device.cmd();
    beginPass(GpuPass::Opaque);
    cmd.beginTarget(mainChain.scene, sceneClear, 0.0f); // HDR scene target, reversed-Z: far is 0
    cmd.setViewport(sceneViewport[0] * sceneScale, sceneViewport[1] * sceneScale, std::max(1.0f, sceneViewport[2] * sceneScale),
                    std::max(1.0f, sceneViewport[3] * sceneScale));
    if (sceneSky) drawSky(sceneParams);
    cmd.setPipeline(scenePipeline());
}

// One point through a 4×4 matrix (row vector convention, divided by w)
static Vec3 transformPoint(const f32 m[16], Vec3 v) {
    const f32 x = v.x * m[0] + v.y * m[4] + v.z * m[8] + m[12];
    const f32 y = v.x * m[1] + v.y * m[5] + v.z * m[9] + m[13];
    const f32 z = v.x * m[2] + v.y * m[6] + v.z * m[10] + m[14];
    const f32 w = v.x * m[3] + v.y * m[7] + v.z * m[11] + m[15];
    return Vec3{x, y, z} * (1.0f / w);
}

// Cascade splitting (docs/DECISIONS.md 2026-10-27 shadows): a bounding sphere per camera slice → orthographic along the light. The sphere radius and center are snapped
// to texels so shadow edges don't shimmer when the camera rotates or moves. 100 m more toward the light so off-screen objects' shadows are received too
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
    const Vec3 ahead = normalize(transformPoint(inv, {0.0f, 0.0f, 1.0f}) - eye); // center of the near plane (reversed-Z)
    Vec3 rays[4];
    const f32 corners[4][2] = {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    for (u32 i = 0; i < 4; ++i) {
        const Vec3 d = normalize(transformPoint(inv, {corners[i][0], corners[i][1], 1.0f}) - eye);
        rays[i] = d * (1.0f / std::max(dot(d, ahead), 1e-3f)); // movement per unit of forward distance
    }
    const f32 splits[kMaxShadowCascades + 1] = {0.0f, cascades == 1 ? s.distance : s.distance * 0.25f, s.distance};
    const Vec3 toSun = normalize(p.sunDirection);
    const Vec3 up = std::abs(toSun.y) > 0.99f ? Vec3{0, 0, 1} : Vec3{0, 1, 0};
    const Mat4 lightView = lookAtLH({}, toSun * -1.0f, up); // looks along the light direction from the origin (position decided by the ortho box)
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
        Mat4 ortho; // orthographic (reversed-Z: near plane 1, far plane 0)
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
    for (u32 c = 0; c < m->shadowCascades; ++c) {
        cmd.setViewport(size * static_cast<f32>(c), 0.0f, size, size);
        const Frustum frustum = Frustum::fromViewProj(m->shadowViewProj[c]);
        const auto& buckets = list.buckets();
        const size_t meshCount = std::min(buckets.size(), m->meshes.size());
        cullInstances(buckets, meshCount, frustum, [&](size_t mesh) { return m->meshes[mesh].bounds; }, m->shadowCull);
        for (size_t mesh = 0; mesh < meshCount; ++mesh) {
            const std::vector<InstanceData>& visible = m->shadowCull.visible[mesh];
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
            if (gm.skinned) continue; // skinned meshes below together with palettes
            ForwardPush push{m->device.uploadBufferIndex(), static_cast<u32>(m->shadowFrameOffsets[c]), static_cast<u32>(offset), 0, 0};
            gm.setBox(push);
            cmd.pushConstants(&push, sizeof(push));
            cmd.setVertexBuffer(gm.vertexBuffer, gm.stride());
            cmd.setIndexBuffer(gm.indexBuffer);
            cmd.drawIndexed(gm.indexCount, static_cast<u32>(visible.size()));
            m->stats.drawCalls += 1;
        }
        // Skinned meshes: upload the palette per character, one at a time
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
            gm.setBox(push);
            cmd.pushConstants(&push, sizeof(push));
            cmd.setVertexBuffer(gm.vertexBuffer, gm.stride());
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
    rhi::CommandList& cmd = device.cmd(); // without an environment the shader draws the fog → sky color gradient
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

// Dynamic resolution controller: GPU time lags 2 frames so it's averaged, and after a change it waits a few frames for the result.
// When over, assumes time is proportional to scale² (pixel count) and corrects in one step; with plenty of headroom it raises slowly (a 75~95% band prevents oscillation)
void Renderer::Impl::updateResolution() {
    const ResolutionSettings& r = resolution;
    const f32 top = std::clamp(r.maxScale, 0.25f, 1.0f), bottom = std::clamp(std::min(r.minScale, top), 0.25f, 1.0f);
    if (!r.dynamic) {
        renderScale = top;
        return;
    }
    renderScale = std::clamp(renderScale, bottom, top);
    const rhi::GpuTimes& gpu = device.gpuTimes();
    if (!gpu.valid || gpu.frameMs <= 0.0f || r.budgetMs <= 0.0f) return;
    gpuAverage = gpuAverage > 0.0f ? gpuAverage * 0.8f + gpu.frameMs * 0.2f : gpu.frameMs;
    if (scaleCooldown) {
        --scaleCooldown;
        return;
    }
    f32 next = renderScale;
    if (gpuAverage > r.budgetMs * 0.95f) next = renderScale * std::sqrt(r.budgetMs * 0.85f / gpuAverage);
    else if (gpuAverage < r.budgetMs * 0.75f) next = renderScale + 0.02f;
    next = std::clamp(std::round(next * 64.0f) / 64.0f, bottom, top); // in 1/64 steps (small jitter is dropped)
    if (next != renderScale) {
        renderScale = next;
        gpuAverage = 0.0f;
        scaleCooldown = 8;
    }
}

void Renderer::setDynamicResolution(bool dynamic, f32 minScale, f32 budgetMs, f32 sharpness) {
    if (!m) return;
    m->resolution.dynamic = dynamic;
    m->resolution.minScale = minScale;
    m->resolution.budgetMs = budgetMs;
    m->resolution.sharpness = sharpness;
}

void Renderer::setRenderScale(f32 maxScale) {
    if (m) m->resolution.maxScale = std::clamp(maxScale, 0.25f, 1.0f);
}

const ResolutionSettings& Renderer::resolution() const {
    static const ResolutionSettings none{};
    return m ? m->resolution : none;
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
                             f32 time, f32 scale) {
    rhi::CommandList& cmd = device.cmd();
    if (timed) beginPass(GpuPass::Post);
    const f32 w = static_cast<f32>(chain.width), h = static_cast<f32>(chain.height);
    // Scene region uv (reduced area). The uber pass reads output pixels × scale / size
    const f32 region[4] = {rect[0] * scale / w, rect[1] * scale / h, (rect[0] + std::max(1.0f, rect[2] * scale)) / w,
                           (rect[1] + std::max(1.0f, rect[3] * scale)) / h};
    const bool sharpen = scale < 0.999f && resolution.sharpness > 0.0f;
    const bool tone = post.toneMapping;
    const bool bloom = tone && postOptions.bloom && post.bloomIntensity > 0.0f;
    const u32 lut = post.lut < luts.size() ? post.lut : 0;
    if (bloom) ensureBloom(chain); // on first use (creation is fine inside a frame too: same as shadow maps)
    const bool bloomReady = bloom && chain.bloom[0];

    PostConstants pc{};
    const f32 exposure[4] = {std::exp2(post.exposure) * postOptions.brightness, std::max(post.bloomIntensity, 0.0f),
                             std::clamp(post.vignette, 0.0f, 1.0f), std::clamp(post.vignetteSmoothness, 0.0f, 1.0f)};
    const f32 effects[4] = {postOptions.filmGrain ? std::clamp(post.grain, 0.0f, 1.0f) : 0.0f,
                            postOptions.chromaticAberration ? std::clamp(post.chromaticAberration, 0.0f, 0.1f) : 0.0f,
                            std::fmod(time, 64.0f), std::clamp(resolution.sharpness, 0.0f, 1.0f)};
    const u32 flags = (tone ? kPostToneMap | kPostDither | (postOptions.antiAliasing ? kPostFxaa : 0u) | (post.punchy ? kPostPunchy : 0u) : 0u) |
                      (sharpen ? kPostSharpen : 0u);
    const u32 postTextures[4] = {bloomReady ? device.textureIndex(device.renderTargetTexture(chain.bloom[0])) : kNoTexture,
                             tone && (lut != 0 || postOptions.colorBlind != ColorBlindMode::Off) ? device.textureIndex(luts[lut].texture)
                                                                                                 : kNoTexture,
                             flags, 0};
    const f32 bloomColor[4] = {post.bloomTint.x, post.bloomTint.y, post.bloomTint.z, std::max(post.bloomThreshold, 0.0f)};
    const f32 sceneSize[4] = {w, h, scale / w, scale / h};
    std::memcpy(pc.exposure, exposure, sizeof(exposure));
    if (tone) std::memcpy(pc.effects, effects, sizeof(effects));
    pc.effects[3] = effects[3];
    std::memcpy(pc.textures, postTextures, sizeof(postTextures));
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

    if (bloomReady) { // dual filter: halve going down (the first step thresholds) and come back up with additive blend
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
    if (debugView != DebugView::None) post.toneMapping = false; // view modes keep colors as is
    runPost(mainChain, post, sceneViewport, {}, true, sceneParams.time, sceneScale);
}

rhi::RenderTarget Renderer::createRenderTarget(u32 width, u32 height) {
    if (!m || width == 0 || height == 0) return {};
    Impl::Offscreen o;
    o.output = m->device.createRenderTarget(width, height, rhi::TargetKind::Ldr);
    m->resizeChain(o.chain, width, height); // paired HDR scene target
    if (m->offscreen.size() <= o.output.id) m->offscreen.resize(o.output.id + 1);
    m->offscreen[o.output.id] = o;
    return o.output;
}

rhi::Texture Renderer::renderTargetTexture(rhi::RenderTarget target) const {
    return m ? m->device.renderTargetTexture(target) : rhi::Texture{};
}

void Renderer::drawToTarget(rhi::RenderTarget target, const FrameParams& p, const DrawList& list, LightingModel lighting) {
    if (!m || !m->inFrame || !target || target.id >= m->offscreen.size() || m->offscreen[target.id].output.id != target.id) return;
    m->ensureScene(); // first, if the main view hasn't been cleared yet
    m->resolveScene(); // the main view goes through post to the back buffer here (no main view 3D after this)
    Impl::PostChain& chain = m->offscreen[target.id].chain;
    rhi::CommandList& cmd = m->device.cmd();
    // Temporarily changes the main view state: frame constants, frustum, view mode (offscreen is always the lit image)
    const u64 frameOffset = m->frameOffset;
    const Frustum frustum = m->frustum;
    const DebugView debugView = m->debugView;
    const LightingModel projectLighting = m->lighting;
    std::vector<PointLight> frameLights = m->frameLights;
    m->debugView = DebugView::None;
    m->lighting = lighting;
    m->writeFrameConstants(p, &m->frameOffset, false); // offscreen without shadows (lookdev, reflection probe baking)
    m->frustum = Frustum::fromViewProj(p.viewProj);
    m->cameraPos = p.cameraPos;
    m->cameraForward = p.cameraForward;

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
    if (m->meshes[meshId].skinned) { // a skinned mesh added without a palette: bind pose (identity palette)
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
    const f32 boundsScale = mesh.skinned ? DrawList::kSkinnedBoundsScale : 1.0f; // light selection (animation margin)
    u64 lightOffset = 0;
    if (!frameLights.empty()) { // point lights: pick the most influential lights per instance (lights touching the bounding sphere, by brightness at the center)
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
                const f32 near = std::max(dist - radius, 0.0f); // the point closest to the object
                const f32 score = (light.color.x + light.color.y + light.color.z) * (1.0f - near / light.range) / (near * near + 1.0f);
                u32 at = picked < lightsPerObject ? picked++ : lightsPerObject;
                if (at == lightsPerObject) { // full: replace the weakest if this one is stronger
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
    mesh.setBox(push);
    if (mesh.skinned) cmd.setPipeline(scenePipeline(true));
    cmd.pushConstants(&push, sizeof(push));
    cmd.setVertexBuffer(mesh.vertexBuffer, mesh.stride());
    cmd.setIndexBuffer(mesh.indexBuffer);
    cmd.drawIndexed(mesh.indexCount, count);
    stats.drawCalls += 1;
    stats.instances += count;
    if (mesh.outline && debugView == DebugView::None) { // outline: the same instances once more with the hull pass
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
    const bool vfxPass = m->openPass == static_cast<i32>(GpuPass::Opaque); // offscreen (lookdev) isn't measured separately
    if (vfxPass) m->beginPass(GpuPass::Vfx);
    cmd.setPipeline(m->particles);
    const ForwardPush push{m->device.uploadBufferIndex(), static_cast<u32>(m->frameOffset), static_cast<u32>(offset), 0, 0};
    cmd.pushConstants(&push, sizeof(push));
    cmd.draw(6, count);
    cmd.setPipeline(m->scenePipeline()); // if more 3D is drawn after (drawToTarget etc.)
    m->stats.drawCalls += 1; // the Vfx section lasts until beginOverlay (UI) (particles are last in DrawList::submit)
}

void Renderer::drawSprites(const Sprite* sprites, u32 count) {
    if (!m || !m->inFrame || !sprites || count == 0) return;
    m->ensureScene();
    count = std::min(count, kMaxSpritesPerFrame);
    const u64 bytes = static_cast<u64>(count) * sizeof(Sprite);
    u64 offset = 0;
    u8* cpu = m->device.allocUpload(bytes, kRingAlignment, &offset);
    if (!cpu) {
        m->warnOverflow(bytes);
        return;
    }
    std::memcpy(cpu, sprites, bytes);
    rhi::CommandList& cmd = m->device.cmd();
    if (m->openPass == static_cast<i32>(GpuPass::Opaque)) m->beginPass(GpuPass::Vfx);
    cmd.setPipeline(m->sprites);
    const ForwardPush push{m->device.uploadBufferIndex(), static_cast<u32>(m->frameOffset), static_cast<u32>(offset), 0, 0};
    cmd.pushConstants(&push, sizeof(push));
    cmd.draw(6, count);
    cmd.setPipeline(m->scenePipeline());
    m->stats.drawCalls += 1;
}

void Renderer::beginOverlay() {
    if (!m || !m->inFrame) return;
    m->ensureScene();
    m->resolveScene();
    m->device.cmd().beginOverlay();
    m->beginPass(GpuPass::Ui); // UI + debug UI
}

void Renderer::updateUiAtlas(UiAtlas atlas, u32 x, u32 y, u32 w, u32 h, const u8* pixels) {
    if (!m || w == 0 || h == 0 || x + w > kUiAtlasSize || y + h > kUiAtlasSize) return;
    const u32 bpp = atlas == UiAtlas::Glyphs ? 1u : 4u;
    // Large regions are split into row batches (so they fit in the frame upload ring)
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
    m->flushUploads(); // copies unbind the draw target: the overlay is rebound below
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
    m->ensureScene(); // clears even frames where nothing was drawn
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
bool Renderer::meshAlphaClip(MeshId mesh) const { return m && mesh < m->meshes.size() && m->meshes[mesh].alphaClip; }
Vec3 Renderer::cameraPos() const { return m ? m->cameraPos : Vec3{}; }
Vec3 Renderer::cameraForward() const { return m ? m->cameraForward : Vec3{0.0f, 0.0f, 1.0f}; }
const Frustum& Renderer::frustum() const {
    static const Frustum none; // with no planes everything is visible
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
    return false; // shipping builds: no view mode pipelines
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
