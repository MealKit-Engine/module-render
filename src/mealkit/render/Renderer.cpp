#include "mealkit/render/Renderer.h"

#include "mealkit/core/Log.h"
#include "mealkit/platform/Image.h"
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
};
static_assert(sizeof(FrameConstants) == 224);

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
    u32 debugView = 0; // render::DebugView (Forward.hlsl PSMain)
};

struct UiPush {
    f32 screenSize[2] = {};
    u32 ring = 0;
    u32 quadOffset = 0;
    u32 glyphAtlas = 0;
    u32 imageAtlas = 0;
    u32 pad[2] = {};
};

constexpr u32 kRingAlignment = 16;

struct GpuMesh {
    rhi::Buffer vertexBuffer;
    rhi::Buffer indexBuffer;
    u32 indexCount = 0;
    BoundingSphere bounds;
    bool outline = false; // 외곽선 머티리얼이 하나라도 있다: 외곽선 패스로 한 번 더 그린다
};

} // namespace

struct Renderer::Impl {
    rhi::Device device;
    rhi::Pipeline forward[kLightingModelCount]; // 조명 모델마다 (init이 모두 만든다: 바꿔도 끊기지 않는다)
    rhi::Pipeline outline;                      // 외곽선 패스 (뒤집은 껍질)
    rhi::Pipeline forwardWire, forwardOverdraw; // 에디터 보기 모드 (디버그 UI 빌드만, PBR 셰이더)
    rhi::Pipeline ui;
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

    rhi::Pipeline& scenePipeline() {
        return debugView == DebugView::Wireframe ? forwardWire
               : debugView == DebugView::Overdraw ? forwardOverdraw
               : debugView == DebugView::None     ? forward[static_cast<u32>(lighting)]
                                                  : forward[0]; // 보기 모드는 조명과 무관한 색 (PBR 셰이더 안)
    }

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
    m->shaderReload.add("Ui.hlsl", [this](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
        rhi::PipelineDesc d = m->uiDesc;
        d.vs = vs;
        d.ps = ps;
        return m->device.recreatePipeline(m->ui, d);
    });

    createMaterial({}); // 0번: 기본 흰색 (머티리얼 없는 메시, 상자)
    m->uiAtlas[0] = m->device.createTexture(kUiAtlasSize, kUiAtlasSize, rhi::Format::R8Unorm);
    m->uiAtlas[1] = m->device.createTexture(kUiAtlasSize, kUiAtlasSize, rhi::Format::Rgba8Unorm);
    m->toonRamp = m->device.createTexture(kToonRampWidth, 1, rhi::Format::Rgba8Unorm);
    setToonRamp(nullptr);
    return true;
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
    mesh.vertexBuffer =
        m->device.createBuffer(rhi::BufferUsage::Vertex, data.vertices.data(), data.vertices.size() * sizeof(Vertex));
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
    if (!m || !m->device.beginFrame()) return false;
    rhi::CommandList& cmd = m->device.cmd();

    const Vec3 sky = lerp(p.fogColor, p.tintColor, std::clamp(p.tintAmount, 0.0f, 1.0f)); // 배경도 번쩍임을 받는다
    m->debugView = supportsDebugView(p.debugView) ? p.debugView : DebugView::None;
    const bool overdraw = m->debugView == DebugView::Overdraw;
    const f32 clear[4] = {overdraw ? 0.0f : sky.x, overdraw ? 0.0f : sky.y, overdraw ? 0.0f : sky.z, 1.0f};
    m->flushUploads(); // 툰 램프·UI 아틀라스 (복사는 그리기 대상을 묶기 전에)
    cmd.beginScene(clear, 0.0f); // reversed-Z: 먼 쪽이 0
    m->beginPass(GpuPass::Opaque);

    const f32 fullW = static_cast<f32>(width()), fullH = static_cast<f32>(height());
    f32 vx = 0.0f, vy = 0.0f, vw = fullW, vh = fullH;
    if (p.viewport.width >= 1.0f && p.viewport.height >= 1.0f) {
        vx = std::clamp(p.viewport.x, 0.0f, fullW - 1.0f);
        vy = std::clamp(p.viewport.y, 0.0f, fullH - 1.0f);
        vw = std::min(p.viewport.width, fullW - vx);
        vh = std::min(p.viewport.height, fullH - vy);
    }
    cmd.setViewport(vx, vy, vw, vh);
    cmd.setPipeline(m->scenePipeline());

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
    fc.style[0] = m->device.textureIndex(m->toonRamp);
    u8* cpu = m->device.allocUpload(sizeof(fc), kRingAlignment, &m->frameOffset); // 프레임의 첫 할당이라 항상 자리가 있다
    u64 materialOffset = 0;
    const u64 materialBytes = m->materials.size() * sizeof(MaterialGpu);
    if (u8* mat = m->device.allocUpload(materialBytes, kRingAlignment, &materialOffset)) std::memcpy(mat, m->materials.data(), materialBytes);
    else m->warnOverflow(materialBytes);
    fc.materials[0] = static_cast<u32>(materialOffset);
    fc.materials[1] = static_cast<u32>(m->materials.size());
    std::memcpy(cpu, &fc, sizeof(fc));

    m->stats = {};
    m->frustum = Frustum::fromViewProj(p.viewProj);
    const rhi::GpuTimes& gpu = m->device.gpuTimes();
    m->stats.gpuValid = gpu.valid;
    m->stats.gpuFrameMs = gpu.frameMs;
    for (u32 i = 0; i < kGpuPassCount; ++i) m->stats.gpuPassMs[i] = gpu.timerMs[i];
    m->inFrame = true;
    return true;
}

void Renderer::drawInstances(MeshId meshId, const InstanceData* instances, u32 count) {
    if (!m || !m->inFrame || count == 0 || meshId >= m->meshes.size()) return;

    const u64 bytes = static_cast<u64>(count) * sizeof(InstanceData);
    u64 offset = 0;
    u8* cpu = m->device.allocUpload(bytes, kRingAlignment, &offset);
    if (!cpu) {
        m->warnOverflow(bytes);
        return;
    }
    std::memcpy(cpu, instances, bytes);

    const GpuMesh& mesh = m->meshes[meshId];
    rhi::CommandList& cmd = m->device.cmd();
    const ForwardPush push{m->device.uploadBufferIndex(), static_cast<u32>(m->frameOffset), static_cast<u32>(offset),
                           static_cast<u32>(m->debugView)};
    cmd.pushConstants(&push, sizeof(push));
    cmd.setVertexBuffer(mesh.vertexBuffer, sizeof(Vertex));
    cmd.setIndexBuffer(mesh.indexBuffer);
    cmd.drawIndexed(mesh.indexCount, count);
    m->stats.drawCalls += 1;
    m->stats.instances += count;
    if (mesh.outline && m->debugView == DebugView::None) { // 외곽선: 같은 인스턴스를 껍질 패스로 한 번 더
        cmd.setPipeline(m->outline);
        cmd.pushConstants(&push, sizeof(push));
        cmd.drawIndexed(mesh.indexCount, count);
        cmd.setPipeline(m->scenePipeline());
        m->stats.drawCalls += 1;
    }
}

void Renderer::beginOverlay() {
    if (!m || !m->inFrame) return;
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
    m->endPass();
    if (m->capturePath.empty()) {
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
