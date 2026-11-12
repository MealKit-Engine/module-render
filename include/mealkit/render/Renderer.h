#pragma once

#include "mealkit/render/Culling.h"
#include "mealkit/render/Mesh.h"
#include "mealkit/rhi/Rhi.h"

#include <mkrender/Renderer.h>

#include <memory>
#include <string>

namespace mealkit {
struct Image; // platform/Image.h
}

// Engine glue for mkrender (MkRender, third_party/mkrender): the renderer itself is the engine-free library, fed with the engine's own
// math/mesh/image types (render/MkRenderConfig.h). mealkit::render names it for engine and game code (render::InstanceData, FrameParams ...)
// and adds what needs the engine: the job system for culling, GPU capture tools, shader hot reload and PNG screenshots.
namespace mealkit::render {

using namespace ::mkrender;

struct RendererDesc {
    void* window = nullptr; // HWND
    u32 width = 0;
    u32 height = 0;
    bool vsync = true;
    bool debugLayer = false;
    bool gpuCapture = false;   // --gpu-capture (rhi/GpuCapture.h)
    std::string pipelineCache; // rhi::DeviceDesc::pipelineCache (user folder; empty = off)
};

class ShaderReloader;

// mkrender::Renderer with the engine's services. Everything else (meshes, frames, post, UI ...) is the library's API
class Renderer : public mkrender::Renderer {
public:
    Renderer();
    ~Renderer();

    // Engine log, culling on the job workers (jobs::acquire until shutdown), capture tools under gpuCapture, view mode pipelines in
    // builds with debug UI, shader hot reload in dev builds (render/ShaderReload.h)
    bool init(const RendererDesc& desc);
    void shutdown();
    // Applies finished shader reloads first (outside a frame), then mkrender::Renderer::beginFrame
    bool beginFrame(const FrameParams& params);
    // Saves the back buffer as a PNG at the next endFrame
    void requestCapture(const char* pngPath);
    using mkrender::Renderer::requestCapture;
    // The engine's Image (platform/Image.h) for the toon ramp and offscreen captures; converted to mkrender::Image
    void setToonRamp(const mealkit::Image* ramp);
    void captureTarget(rhi::RenderTarget target, mealkit::Image* out);
    // mkrender::Renderer::endFrame, then hands a finished captureTarget to its Image
    void endFrame();

private:
    std::unique_ptr<ShaderReloader> m_shaderReload;
    std::string m_capturePath;
    mkrender::Image m_targetCapture;
    mealkit::Image* m_targetOut = nullptr;
    bool m_jobs = false;
};

} // namespace mealkit::render
