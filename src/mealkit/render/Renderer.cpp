#include "mealkit/render/Renderer.h"

#include "mealkit/core/Log.h"
#include "mealkit/platform/Image.h"
#include "mealkit/render/ShaderReload.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "g_ForwardPS.h"
#include "g_ForwardVS.h"
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
};
static_assert(sizeof(FrameConstants) == 192);

// shaders/Forward.hlsl Material (64바이트)
struct MaterialGpu {
    f32 baseColor[4];
    f32 emissive[3];
    f32 alphaCutoff;
    f32 metallic, roughness, normalScale, pad;
    u32 textures[4]; // baseColor, normal, metallicRoughness, emissive (bindless 번호, kNoTexture = 없음)
};
static_assert(sizeof(MaterialGpu) == 64);

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
};

} // namespace

struct Renderer::Impl {
    rhi::Device device;
    rhi::Pipeline forward;
    rhi::Pipeline forwardWire, forwardOverdraw; // 에디터 보기 모드 (디버그 UI 빌드만)
    rhi::Pipeline ui;
    DebugView debugView = DebugView::None;
    rhi::PipelineDesc forwardDesc, uiDesc; // 셰이더 핫 리로드가 바이트코드만 바꿔 다시 만든다
    ShaderReloader shaderReload;
    std::vector<GpuMesh> meshes;
    std::vector<MaterialGpu> materials; // 0 = 기본 흰색. 프레임마다 업로드 링에 쓴다
    std::vector<rhi::Texture> textures;

    // 2D UI (shaders/Ui.hlsl): 글자 SDF 아틀라스 + 이미지 아틀라스
    rhi::Texture uiAtlas[2];
    struct AtlasUpload {
        u8 atlas;
        u32 x, y, w, h;
        std::vector<u8> pixels;
    };
    std::vector<AtlasUpload> uiUploads; // 아직 GPU에 올리지 않은 영역 (차례대로)

    u64 frameOffset = 0; // 이번 프레임 FrameConstants의 링 위치
    bool vsync = true;
    bool inFrame = false;
    bool uploadOverflowWarned = false;
    RenderStats stats;
    Frustum frustum;
    i32 openPass = -1; // 지금 GPU 시간을 재는 패스 (GpuPass, 없으면 -1)
    std::string capturePath;

    rhi::PipelineDesc wireDesc() const {
        rhi::PipelineDesc d = forwardDesc;
        d.wireframe = true;
        d.cullBack = false;
        return d;
    }
    rhi::PipelineDesc overdrawDesc() const {
        rhi::PipelineDesc d = forwardDesc;
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

    void flushUiUploads() {
        rhi::CommandList& cmd = device.cmd();
        size_t done = 0;
        for (; done < uiUploads.size(); ++done) {
            const AtlasUpload& u = uiUploads[done];
            const u32 bpp = u.atlas == 0 ? 1u : 4u;
            const u32 pitch = device.copyPitch(u.w, bpp);
            u64 offset = 0;
            u8* dst = device.allocUpload(static_cast<u64>(pitch) * u.h, device.copyAlignment(), &offset);
            if (!dst) break;
            for (u32 row = 0; row < u.h; ++row)
                std::memcpy(dst + static_cast<size_t>(row) * pitch, u.pixels.data() + static_cast<size_t>(row) * u.w * bpp,
                            static_cast<size_t>(u.w) * bpp);
            const rhi::Texture tex = uiAtlas[u.atlas];
            cmd.textureBarrier(tex, rhi::TextureState::ShaderRead, rhi::TextureState::CopyDst);
            cmd.copyToTexture(tex, u.x, u.y, u.w, u.h, offset, pitch);
            cmd.textureBarrier(tex, rhi::TextureState::CopyDst, rhi::TextureState::ShaderRead);
        }
        uiUploads.erase(uiUploads.begin(), uiUploads.begin() + static_cast<std::ptrdiff_t>(done));
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

    rhi::PipelineDesc forward;
    forward.vs = {g_ForwardVS, sizeof(g_ForwardVS)};
    forward.ps = {g_ForwardPS, sizeof(g_ForwardPS)};
    forward.vertexAttributes = {{"POSITION", rhi::VertexFormat::Float3, 0},
                                {"NORMAL", rhi::VertexFormat::Float3, 12},
                                {"COLOR", rhi::VertexFormat::Unorm8x4, 24},
                                {"TEXCOORD", rhi::VertexFormat::Half2, 28},
                                {"MATERIAL", rhi::VertexFormat::Uint16x2, 32}};
    forward.vertexStride = sizeof(Vertex);
    forward.target = rhi::Target::Scene;
    m->forward = m->device.createPipeline(forward);
    m->forwardDesc = forward;
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
    m->shaderReload.add("Forward.hlsl", [this](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
        rhi::PipelineDesc d = m->forwardDesc;
        d.vs = vs;
        d.ps = ps;
        m->forwardDesc = d; // 보기 모드 파이프라인도 같은 셰이더 (바이트코드는 ShaderReloader가 갖고 있다)
#if MEALKIT_DEBUG_UI
        m->device.recreatePipeline(m->forwardWire, m->wireDesc());
        m->device.recreatePipeline(m->forwardOverdraw, m->overdrawDesc());
#endif
        return m->device.recreatePipeline(m->forward, d);
    });
    m->shaderReload.add("Ui.hlsl", [this](const rhi::ShaderCode& vs, const rhi::ShaderCode& ps) {
        rhi::PipelineDesc d = m->uiDesc;
        d.vs = vs;
        d.ps = ps;
        return m->device.recreatePipeline(m->ui, d);
    });

    createMaterial({}); // 0번: 기본 흰색 (머티리얼 없는 메시, 상자)
    m->uiAtlas[0] = m->device.createTexture(kUiAtlasSize, kUiAtlasSize, rhi::Format::R8Unorm);
    m->uiAtlas[1] = m->device.createTexture(kUiAtlasSize, kUiAtlasSize, rhi::Format::Rgba8Unorm);
    return true;
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
    cmd.setPipeline(m->debugView == DebugView::Wireframe  ? m->forwardWire
                    : m->debugView == DebugView::Overdraw ? m->forwardOverdraw
                                                          : m->forward);

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
        Impl::AtlasUpload u;
        u.atlas = static_cast<u8>(atlas);
        u.x = x;
        u.y = y + row;
        u.w = w;
        u.h = std::min(rowsPerPiece, h - row);
        u.pixels.assign(pixels + static_cast<size_t>(row) * w * bpp, pixels + static_cast<size_t>(row + u.h) * w * bpp);
        m->uiUploads.push_back(std::move(u));
    }
}

void Renderer::drawUi(const UiQuad* quads, u32 count) {
    if (!m || !m->inFrame) return;
    m->flushUiUploads(); // 복사는 그리기 대상을 푼다: 아래에서 오버레이를 다시 묶는다
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
