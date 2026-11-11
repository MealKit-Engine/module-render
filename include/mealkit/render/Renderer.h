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
// Mesh, material and texture data are the asset module's (asset/MeshData.h)
using asset::kMaterialDissolve;
using asset::kMaterialOutline;
using asset::kMaterialRim;
using asset::kMaterialWind;
using asset::kNoMesh;
using asset::kNoTexture;
using asset::MaterialData;
using asset::MeshId;

// Lighting model (style preset, docs/CODING_MODEL.md "Shader code"): the single project setting lightingModel decides it for the whole game.
// One shader per model is compiled at build time, and the renderer builds all pipelines in init (switching doesn't hitch)
enum class LightingModel : u8 { Pbr, Toon, Unlit }; // same order as the project setting game::LightingModelSetting
inline constexpr u32 kLightingModelCount = 3;

// Instance data copied to the GPU as is (48 bytes, identical to the HLSL Instance struct).
struct InstanceData {
    Vec3 position;
    u32 color = 0xFFFFFFFF; // sRGB color made with rgba()
    Quat rotation;
    Vec3 scale{1, 1, 1};
    u32 flags = 0;
};
static_assert(sizeof(InstanceData) == 48, "InstanceData must match the HLSL layout");

inline constexpr u32 kInstanceChecker = 1u;   // checker pattern for floors
inline constexpr u32 kInstanceHighlight = 2u; // editor selection highlight
inline constexpr u32 kInstanceOverBudget = 4u; // editor view mode Budget: models over the asset budget
inline constexpr u32 kInstanceLodShift = 8;    // editor view mode Lod: bits 8~9 = drawn LOD level
// Per-instance style values (upper 16 bits of flags, the instance stays 48 bytes): changed by view/ from tick events
inline constexpr u32 kInstanceFlashShift = 16;    // bits 16~23: flash 0..1 (blended toward white, all materials)
inline constexpr u32 kInstanceDissolveShift = 24; // bits 24~31: dissolve progress 0..1 (kMaterialDissolve materials only)
inline u32 instanceUnit8(f32 amount) { return static_cast<u32>((amount < 0.0f ? 0.0f : amount > 1.0f ? 1.0f : amount) * 255.0f + 0.5f); }
// Puts a flash into flags: out.add(mesh, {..., instanceFlash(flags, 0.8f)})
inline u32 instanceFlash(u32 flags, f32 amount) { return (flags & ~(0xFFu << kInstanceFlashShift)) | (instanceUnit8(amount) << kInstanceFlashShift); }
inline u32 instanceDissolve(u32 flags, f32 progress) {
    return (flags & ~(0xFFu << kInstanceDissolveShift)) | (instanceUnit8(progress) << kInstanceDissolveShift);
}

// Editor view modes (what's expensive in the scene view, by color). Pipelines exist only in builds with debug UI; shipping builds have only None
enum class DebugView : u8 {
    None,
    Wireframe, // triangle lines
    Normals,   // normal direction = color
    Lod,       // LOD 0 green, 1 yellow, 2 red
    Overdraw,  // number of times the same pixel was drawn = brightness (additive without depth test)
    Budget,    // models over budget red, the rest gray
    Features,  // material features: outline red, wind green, rim blue, dissolve yellow (summed when combined), none gray
    Count
};

constexpr u32 rgba(u8 r, u8 g, u8 b, u8 a = 255) {
    return static_cast<u32>(r) | (static_cast<u32>(g) << 8) | (static_cast<u32>(b) << 16) | (static_cast<u32>(a) << 24);
}

// One particle (32 bytes like HLSL Particle, shaders/Particle.hlsl). A round quad facing the camera, after opaque 3D,
// depth tested only (no writes) and blended with premultiplied alpha. Additive particles output alpha 0 (one pipeline)
struct Particle {
    Vec3 position;
    f32 size = 0.1f;   // diameter (meters)
    u32 color = 0xFFFFFFFF; // sRGB RGBA8 (rgba()), straight alpha
    f32 rotation = 0.0f; // radians (in screen space)
    u32 flags = 0;       // kParticleAdditive
    f32 softness = 0.5f; // edge blur 0 (hard circle) .. 1 (blurs from the center)
};
static_assert(sizeof(Particle) == 32, "Particle must match the HLSL layout");
inline constexpr u32 kParticleAdditive = 1u;
inline constexpr u32 kMaxParticlesPerFrame = 16384; // beyond this, from the front (upload ring 512 KB)

// One sprite (64 bytes like HLSL Sprite, shaders/Sprite.hlsl): a textured quad in the world on the plane of its rotation
// (identity = facing -z, the 2D camera's view). Drawn after opaque 3D and before particles, depth tested without writes,
// back to front (DrawList::submit sorts by layer, then distance). The texture holds premultiplied alpha (render/Sprites.h).
struct Sprite {
    Vec3 position;           // where the pivot lands
    u32 texture = 0;         // bindless index from SpriteTextures (render/Sprites.h)
    Quat rotation;           // the quad's x/y plane
    f32 width = 1.0f, height = 1.0f; // meters
    f32 pivotX = 0.5f, pivotY = 0.5f; // 0..1 from the image's left/bottom
    f32 u0 = 0.0f, v0 = 0.0f, u1 = 1.0f, v1 = 1.0f; // image rect (v down from the top), for sheet cells
    u32 color = 0xFFFFFFFF;  // tint, sRGB RGBA8 (rgba()), straight alpha
    u32 flags = 0;           // kSprite*
    i32 layer = 0;           // draw order before distance: higher layers draw over lower ones
    f32 pad = 0.0f;
};
static_assert(sizeof(Sprite) == 80, "Sprite must match the HLSL layout");
inline constexpr u32 kSpriteFlipX = 1u;
inline constexpr u32 kSpriteFlipY = 2u;
inline constexpr u32 kSpritePoint = 4u; // nearest texel (pixel art) instead of linear filtering
inline constexpr u32 kMaxSpritesPerFrame = 8192; // beyond this, the last in draw order are dropped (640 KB of the upload ring)

// Screen area (pixels) to draw 3D in. Whole window if width is 0.
struct Viewport {
    f32 x = 0.0f, y = 0.0f, width = 0.0f, height = 0.0f;
};

// One reflection probe (a scene reflection_probe object, filled by render::probes (render/ModelDraw.h)). An object uses the reflection cube of the one probe
// whose influence box contains its position with the closest center (no blending). If in no box, the sky HDRI reflection
struct ReflectionProbe {
    Vec3 position;     // baked position = influence box center
    Vec3 halfExtents;  // influence box (axis-aligned, world)
    u32 id = 0; // Renderer::createProbe index (0 = none, skipped)
};
inline constexpr u32 kMaxReflectionProbes = 64; // the shader scans them all per object (per-scene limit is the project setting probeMaxCount)

// One point light (a scene point_light object = render::addSceneLights; brief lights from view-side effects are added by view/). No shadows.
// The renderer picks up to setLightsPerObject most influential lights per object (instance) on the CPU and passes their indices to the shader
struct PointLight {
    Vec3 position;
    f32 range = 6.0f;   // becomes 0 at this distance (meters)
    Vec3 color{1, 1, 1}; // linear color × intensity
};
inline constexpr u32 kMaxPointLights = 255;      // per frame (8-bit index, beyond this from the front)
inline constexpr u32 kMaxLightsPerObject = 8;    // 8 indices per instance (8 bytes)

// Sun shadows (CSM, docs/ROADMAP.md stage 2 "Shadows"): splits up to distance in front of the camera into 1~2 cascades with a size² shadow map each.
// Decided by project settings shadows* (game::applyRenderStyle). Point lights have no shadows
struct ShadowSettings {
    bool enabled = true;
    u32 cascades = 2;     // 1 or 2
    u32 size = 1024;      // shadow map size of one cascade (512..4096)
    f32 distance = 40.0f; // shadows up to this distance (meters) from the camera (fading over the last 10%)
};
inline constexpr u32 kMaxShadowCascades = 2;

// Post-processing (docs/ROADMAP.md stage 2 "Post-processing", shaders/Post.hlsl): the game/editor fill in the scene environment's values (scene::PostFields).
// View values, unrelated to determinism. The scene is drawn in HDR (linear) and one uber pass tone maps it to the back buffer
struct PostParams {
    bool toneMapping = true; // false: linear values are only clamped to sRGB (reflection probe baking). Bloom/LUT/effects are off too
    bool punchy = true;      // AgX "Punchy" look: more contrast and saturation than the flat base look
    f32 exposure = 0.0f;     // EV (2x = +1)
    u32 lut = 0;             // color grading LUT (createLut index, 0 = none)
    f32 bloomIntensity = 0.25f; // 0 = skip the bloom pass
    f32 bloomThreshold = 1.0f;  // only light brighter than this (linear, before exposure)
    Vec3 bloomTint{1.0f, 1.0f, 1.0f};
    f32 vignette = 0.0f;           // 0..1 darken edges
    f32 vignetteSmoothness = 0.5f; // start radius 0..1
    f32 grain = 0.0f;              // film grain 0..1
    f32 chromaticAberration = 0.0f; // chromatic aberration (fraction of region size, 0.01 = 1% at the edge)
};

// Player settings (settings::DisplaySettings/GameplaySettings): disabling effects, brightness, color blindness correction. Applied before scene values
enum class ColorBlindMode : u8 { Off, Protanopia, Deuteranopia, Tritanopia }; // same order as settings::ColorBlindSetting
struct PostOptions {
    bool bloom = true;
    bool antiAliasing = true; // FXAA (inside the uber pass)
    bool filmGrain = true;
    bool chromaticAberration = true;
    f32 brightness = 1.0f; // multiplied into exposure
    ColorBlindMode colorBlind = ColorBlindMode::Off; // layered on top of the LUT (the CPU combines them into one LUT)
};
inline constexpr u32 kLutSize = 32;

// Dynamic resolution (docs/ROADMAP.md stage 2 "Dynamic resolution"): the main view 3D is drawn into a reduced area of the HDR target and the uber pass upscales it to native resolution
// (bilinear + contrast adaptive sharpening). UI is always native resolution. The controller lowers the scale when GPU frame time exceeds budgetMs
// and raises it slowly up to maxScale when there's headroom. Offscreen (drawToTarget) and view modes are always 1
struct ResolutionSettings {
    bool dynamic = false;  // project setting dynamicResolution (fixed at maxScale when off)
    f32 minScale = 0.5f;   // project setting dynamicResolutionMinScale
    f32 maxScale = 1.0f;   // player setting render scale (setRenderScale)
    f32 budgetMs = 8.0f;   // sum of the project GPU budget
    f32 sharpness = 0.5f;  // sharpening when reduced 0..1
}; // side length of the color grading LUT (.cube files are resampled to this size)

struct FrameParams {
    Viewport viewport;
    Mat4 viewProj;
    Vec3 cameraPos;
    Vec3 cameraForward{0.0f, 0.0f, 1.0f}; // sound listener direction (the engine passes it to audio::Engine::setListener every frame)
    Vec3 sunDirection{0.4f, 0.8f, 0.3f}; // direction toward the sun (the renderer normalizes it)
    Vec3 sunColor{1.0f, 0.95f, 0.85f};
    Vec3 skyColor{0.18f, 0.36f, 0.85f}; // linear color space; without an environment also the sky overhead
    Vec3 groundColor{0.20f, 0.17f, 0.14f};
    Vec3 fogColor{0.55f, 0.70f, 0.95f}; // also the background clear color and, without an environment, the sky at the horizon
    f32 fogDensity = 0.003f;
    // Blends the whole screen toward this color (linear) (flash from the feedback table, filled by the engine from feedback::Player). 0 = none
    Vec3 tintColor{1.0f, 1.0f, 1.0f};
    f32 tintAmount = 0.0f;
    DebugView debugView = DebugView::None; // editor view mode
    // View time (seconds, wind sway). Graphics regression tests keep it at 0
    f32 time = 0.0f;
    Vec3 windDirection{1.0f, 0.0f, 0.3f}; // horizontal direction (the renderer normalizes it)
    f32 windStrength = 1.0f;              // multiplied into material windStrength (0 = no wind)
    // Sky HDRI (Renderer::createEnvironment index, 0 = none: sky/ground color gradient and fog color background). If present, the skybox is drawn and
    // ambient (SH) and reflections (cube mips = roughness) are read from that HDRI. Intensity multiplies sky/ambient/reflections, rotation is about the y axis (radians)
    u32 environment = 0;
    f32 environmentIntensity = 1.0f;
    f32 environmentRotation = 0.0f;
    // Reflection probes (caller's memory, read only during beginFrame/drawToTarget). Beyond kMaxReflectionProbes, from the front
    const ReflectionProbe* probes = nullptr;
    u32 probeCount = 0;
    // Point lights (caller's memory, read during the frame). Beyond kMaxPointLights, from the front
    const PointLight* lights = nullptr;
    u32 lightCount = 0;
    PostParams post; // not used in view modes (colors as is)
};

// Environment baked from a sky HDRI (render::EnvironmentMap, cubes in RGB9E5, faces +X −X +Y −Y +Z −Z in order). createEnvironment uploads it to the GPU
struct EnvironmentData {
    f32 sh[9][3] = {}; // ambient color = Σ sh[i] · Y_i(n)
    u32 skySize = 0;
    const u32* sky = nullptr; // skySize² per face
    u32 reflectSize = 0, reflectMips = 0;
    const u32* reflect = nullptr; // per face from mip 0
};

struct RendererDesc {
    void* window = nullptr; // HWND
    u32 width = 0;
    u32 height = 0;
    bool vsync = true;
    bool debugLayer = false;
    bool gpuCapture = false; // --gpu-capture (rhi/GpuCapture.h)
    std::string pipelineCache; // rhi::DeviceDesc::pipelineCache (user folder; empty = off)
};

// GPU inspection is done by the RHI backend (shipping exe --system-info, min-spec check)
using GpuInfo = rhi::GpuInfo;
using rhi::gpuMinimumError;
using rhi::queryGpu;

// GPU passes: sections that split frame GPU time for measurement (GPU budget table in the benchmark report, mealkit.project gpuBudget*).
// Passes not yet present (shadow/post/VFX) start being measured by the renderer when those features arrive (0 until then)
enum class GpuPass : u8 { Opaque, Shadow, Post, Vfx, Ui, Count };
inline constexpr u32 kGpuPassCount = static_cast<u32>(GpuPass::Count);
const char* gpuPassName(GpuPass pass); // "opaque", "shadow", ... (reports/logs, English)

struct RenderStats {
    u32 drawCalls = 0;
    u32 instances = 0;
    u32 culled = 0; // instances dropped for being outside the frustum (DrawList::submit)
    // GPU time (timestamps): values from the most recent frame the GPU finished, so 2 frames late. Unknown if gpuValid is false
    bool gpuValid = false;
    f32 gpuFrameMs = 0.0f;
    f32 gpuPassMs[kGpuPassCount] = {};
    f32 resolutionScale = 1.0f; // main view 3D scale this frame (dynamic resolution)
};

// One 2D UI rectangle (112 bytes like HLSL UiQuad, shaders/Ui.hlsl). Filled by mealkit/ui/. Coordinates in back buffer pixels.
struct UiQuad {
    f32 rect[4] = {};                    // x, y, w, h (box before rotation)
    f32 clip[4] = {-1e9f, -1e9f, 1e9f, 1e9f}; // x0, y0, x1, y1 (nothing drawn outside)
    f32 radii[4] = {};                   // top left, top right, bottom right, bottom left
    f32 uv[4] = {};                      // atlas u0, v0, u1, v1 (images/text)
    f32 gradient[4] = {};                // linear: start xy, end xy / radial: center xy, radius xy (box 0..1 coordinates)
    u32 color0 = 0;                      // sRGB RGBA8 (rgba()), straight alpha
    u32 color1 = 0;
    u32 strokeColor = 0;
    u32 flags = 0; // kUiShape/Image/Glyph/Shadow | kUiLinear/kUiRadial
    f32 strokeWidth = 0.0f;
    f32 rotation = 0.0f; // radians, around the box center
    f32 softness = 0.0f; // shadow blur (pixels)
    f32 sdfScale = 0.0f; // text: screen pixels per SDF value unit
};
static_assert(sizeof(UiQuad) == 112, "UiQuad must match the HLSL layout");

inline constexpr u32 kUiShape = 0, kUiImage = 1, kUiGlyph = 2, kUiShadow = 3;
inline constexpr u32 kUiLinear = 1u << 4, kUiRadial = 2u << 4;

enum class UiAtlas : u8 { Glyphs, Images }; // glyph SDF (1 byte), UI images (RGBA 4 bytes)
inline constexpr u32 kUiAtlasSize = 2048;

// Forward renderer (on top of the RHI, doesn't call graphics APIs directly). Processes 2 frames concurrently, and per-frame data
// is written to the RHI upload ring. On integrated graphics (UMA) the upload memory is the memory the GPU reads, so there's no copy at all.
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
    // New contents for an existing mesh (same id; old buffers deleted after waiting for the GPU). Outside a frame (editor terrain brushes)
    void replaceMesh(MeshId id, const MeshData& mesh);
    // Referenced by vertex material indices. Index 0 is default white (created by init)
    u32 createMaterial(const MaterialData& material);
    // Texture with mips (render/TextureCache.h). The bindless index read by shaders
    u32 createTexture(const rhi::TextureInit& init);
    // Mesh local bounding sphere (createMesh computes it from vertices). Radius 0 for missing meshes
    BoundingSphere meshBounds(MeshId mesh) const;
    // An alpha mask or dissolve material in the mesh (DrawList::submit draws these buckets last)
    bool meshAlphaClip(MeshId mesh) const;
    Vec3 cameraPos() const; // this frame's FrameParams::cameraPos (front-to-back order)
    Vec3 cameraForward() const; // this frame's FrameParams::cameraForward (sprite back-to-front order)
    // Frustum of this frame's viewProj (made by beginFrame). DrawList::submit filters instances with it
    const Frustum& frustum() const;
    void addCulled(u32 count); // for stats (DrawList::submit)

    // Uploads a sky HDRI environment to the GPU and returns an index (FrameParams::environment). Not deleted: the user (ModelCache) reuses the index for the same HDRI
    u32 createEnvironment(const EnvironmentData& data);
    // Uploads a reflection probe cube (render::ProbeMap: RGB9E5, per face from mip 0) and returns an index (ReflectionProbe::id). Not deleted
    u32 createProbe(u32 size, u32 mips, const u32* reflect);
    // Lighting model (project setting, game::applyRenderStyle). All pipelines are prebuilt by init, so it switches immediately
    void setLightingModel(LightingModel model);
    LightingModel lightingModel() const;
    // Toon ramp: colors from the left (dark side) → right (lit side) of a horizontal image multiplied into diffuse lighting (only the first row is used). nullptr = engine default 3 steps
    void setToonRamp(const Image* ramp);
    // Point lights used per object (project setting pointLightsPerObject, 1..kMaxLightsPerObject, default 4)
    void setLightsPerObject(u32 count);
    // Sun shadows (project setting). When size/cascade count change, a new shadow map is made (the old one remains)
    void setShadows(const ShadowSettings& shadows);
    const ShadowSettings& shadows() const;
    // Shadow pass: DrawList::submit calls it before the main view 3D (once per frame; does nothing if shadows are off).
    // All instances before frustum culling cast shadows (filtered per cascade by the light-space box)
    void drawShadows(const DrawList& list);

    // If false, this frame isn't drawn (minimized etc.).
    bool beginFrame(const FrameParams& params);
    // Draws instances of the same mesh in one draw call.
    void drawInstances(MeshId mesh, const InstanceData* instances, u32 count);
    // One skinned mesh (one draw call per character): bones is the palette from asset::skinPalette (joint count, max 64)
    void drawSkinned(MeshId mesh, const InstanceData& instance, const BoneMatrix* bones, u32 boneCount);
    bool meshSkinned(MeshId mesh) const;
    // Particles (DrawList::submit calls it after opaque 3D): one draw call, GPU pass Vfx
    void drawParticles(const Particle* particles, u32 count);
    // Sprites in draw order (DrawList::submit sorts them and calls this after opaque 3D, before particles): one draw call, GPU pass Vfx
    void drawSprites(const Sprite* sprites, u32 count);
    // Color grading LUT (render::loadCubeLut): rgb sRGB values of kLutSize³ cells (red changes fastest). The index is PostParams::lut.
    // Not deleted: the user (ModelCache) reuses the index for the same file
    u32 createLut(const f32* rgb);
    void setPostOptions(const PostOptions& options); // player settings (rebuilds LUTs when color blindness correction changes)
    const PostOptions& postOptions() const;
    // Dynamic resolution: project-side values (dynamic/minScale/budgetMs/sharpness, game::applyRenderStyle) and the player render scale (settings) are set separately
    void setDynamicResolution(bool dynamic, f32 minScale, f32 budgetMs, f32 sharpness);
    void setRenderScale(f32 maxScale); // 0.5..1
    const ResolutionSettings& resolution() const;
    // Offscreen rendering (editor lookdev window, reflection probe baking): makes an 8-bit sRGB target paired with an HDR scene target of the same size.
    // Made once with a fixed size and reused (not deleted)
    rhi::RenderTarget createRenderTarget(u32 width, u32 height);
    rhi::Texture renderTargetTexture(rhi::RenderTarget target) const; // ImGui::Image (rhi::imguiTextureId)
    // After the main view 3D (DrawList::submit), before UI: draws list over the whole target with params' camera/lights/post and the lighting model.
    // The main view goes through post to the back buffer here, so no main view 3D is drawn after this (beginOverlay rebinds for UI)
    void drawToTarget(rhi::RenderTarget target, const FrameParams& params, const DrawList& list, LightingModel lighting);
    // Binds the back buffer to draw UI after 3D (the main view 3D goes through post to the back buffer here)
    void beginOverlay();
    // Changes one region of a UI atlas (anytime; uploaded to the GPU before the next drawUi). pixels are w×h with no gaps between rows
    void updateUiAtlas(UiAtlas atlas, u32 x, u32 y, u32 w, u32 h, const u8* pixels);
    // Draws 2D UI: after 3D, at native back buffer resolution (independent of dynamic resolution), no depth. One draw call
    void drawUi(const UiQuad* quads, u32 count);
    void endFrame();
    void waitIdle();

    rhi::Device& device(); // debug UI drawing (rhi::imgui*)
    u32 width() const;
    u32 height() const;

    void setVsync(bool enabled);
    bool vsync() const;
    void requestCapture(const char* pngPath); // saves the back buffer as a PNG at the next endFrame
    // At this frame's endFrame, reads the offscreen target (drawn with drawToTarget) into out (sRGB 8-bit, waits for the GPU).
    // Reflection probe baking. out must live until endFrame. requestCapture in the same frame is deferred to the next frame
    void captureTarget(rhi::RenderTarget target, Image* out);

    const char* adapterName() const;
    const GpuInfo& gpu() const;
    const char* initError() const; // reason when init() returns false
    bool isUma() const;
    const RenderStats& stats() const;
    bool supportsDebugView(DebugView view) const; // whether this build/device can draw that view mode
    rhi::GpuMemory gpuMemory() const; // a driver query, so not called every frame (FrameStats does it every report period)

private:
    struct Impl;
    Impl* m = nullptr;
};

} // namespace mealkit::render
