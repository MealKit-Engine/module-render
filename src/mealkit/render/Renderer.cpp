#include "mealkit/render/Renderer.h"

#include "mealkit/core/Log.h"
#include "mealkit/jobs/Jobs.h"
#include "mealkit/platform/Image.h"
#include "mealkit/render/ShaderReload.h"
#include "mealkit/rhi/GpuCapture.h"

namespace mealkit::render {
namespace {

void jobsParallelFor(u32 count, u32 minChunk, const std::function<void(u32 begin, u32 end)>& body) {
    jobs::parallelFor(count, minChunk, body);
}

void saveCapture(const mkrender::Image& captured, void* user) {
    std::string& path = *static_cast<std::string*>(user);
    mealkit::Image image;
    image.width = captured.width;
    image.height = captured.height;
    image.rgba = captured.rgba;
    if (image.width && savePng(path, image))
        MEALKIT_INFO("Screenshot saved: %s (%ux%u)", path.c_str(), image.width, image.height);
    else
        MEALKIT_WARN("Screenshot failed: %s", path.c_str());
    path.clear();
}

} // namespace

Renderer::Renderer() = default;
Renderer::~Renderer() { shutdown(); }

bool Renderer::init(const RendererDesc& desc) {
    rhi::connectLog();
    jobs::acquire(); // culling workers (shared with physics); released in shutdown
    m_jobs = true;
    m_shaderReload = std::make_unique<ShaderReloader>();
    mkrender::RendererDesc d;
    d.window = desc.window;
    d.width = desc.width;
    d.height = desc.height;
    d.vsync = desc.vsync;
    d.debugLayer = desc.debugLayer;
    if (desc.gpuCapture) d.beforeDeviceCreate = &rhi::loadGpuCaptureTool;
    d.pipelineCache = desc.pipelineCache;
#if MEALKIT_DEBUG_UI
    d.debugViews = true; // editor view modes; shipping builds have only DebugView::None
#endif
    d.parallelFor = &jobsParallelFor;
    if (m_shaderReload->enabled()) {
        d.shaderReload = [this](const char* file, const char* vs, const char* ps, const char* defines, ShaderApply apply) {
            m_shaderReload->add(file, std::move(apply), vs, ps, defines);
        };
    }
    return mkrender::Renderer::init(d);
}

void Renderer::shutdown() {
    mkrender::Renderer::shutdown();
    m_shaderReload.reset();
    if (m_jobs) {
        jobs::release();
        m_jobs = false;
    }
}

bool Renderer::beginFrame(const FrameParams& params) {
    if (m_shaderReload) m_shaderReload->update(); // outside a frame (waits for the GPU when swapping pipelines)
    return mkrender::Renderer::beginFrame(params);
}

void Renderer::requestCapture(const char* pngPath) {
    m_capturePath = pngPath;
    mkrender::Renderer::requestCapture(&saveCapture, &m_capturePath);
}

void Renderer::setToonRamp(const mealkit::Image* ramp) {
    if (!ramp) {
        mkrender::Renderer::setToonRamp(nullptr);
        return;
    }
    mkrender::Image copy;
    copy.width = ramp->width;
    copy.height = ramp->height;
    copy.rgba = ramp->rgba;
    mkrender::Renderer::setToonRamp(&copy);
}

void Renderer::captureTarget(rhi::RenderTarget target, mealkit::Image* out) {
    if (!out) return;
    m_targetOut = out;
    mkrender::Renderer::captureTarget(target, &m_targetCapture);
}

void Renderer::endFrame() {
    mkrender::Renderer::endFrame();
    if (m_targetOut && m_targetCapture.width) {
        m_targetOut->width = m_targetCapture.width;
        m_targetOut->height = m_targetCapture.height;
        m_targetOut->rgba = std::move(m_targetCapture.rgba);
        m_targetCapture = {};
        m_targetOut = nullptr;
    }
}

} // namespace mealkit::render
