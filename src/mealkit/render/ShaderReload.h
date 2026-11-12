#pragma once

#include "mealkit/rhi/Rhi.h"

#include <functional>
#include <memory>
#include <string>

// Shader hot reload (for engine development, docs/ROADMAP.md stage 2 "Shader hot reload"). In dev builds (MEALKIT_SHADER_HOT_RELOAD,
// builds with debug UI), when .hlsl/Bindless.hlsli in MkRender's shaders/ change (third_party/mkrender/shaders, or the checkout given with MEALKIT_MKRENDER_DIR), recompiles with the same DXC and options as the build
// and swaps the pipelines. Compile errors are logged and the previous shader keeps being used. Shipping builds (build.bat Export) use only embedded
// bytecode and this does nothing. Also off on PCs without that folder or DXC (a copied dev exe).
namespace mealkit::render {

class ShaderReloader {
public:
    // Receives bytecode and rebuilds the pipeline. false on failure (keeps using the previous one)
    using Apply = std::function<bool(const rhi::ShaderCode& vs, const rhi::ShaderCode& ps)>;

    ShaderReloader();
    ~ShaderReloader();
    ShaderReloader(const ShaderReloader&) = delete;
    ShaderReloader& operator=(const ShaderReloader&) = delete;

    // Vertex/pixel entry points of shaders/<file> (default VSMain + PSMain). defines is a single define like "LIGHTING_MODEL=1" (same as the
    // mkrender_shader variants in MkRender's CMakeLists.txt; none if empty)
    void add(const char* file, Apply apply, const char* vsEntry = "VSMain", const char* psEntry = "PSMain",
             const char* defines = "");
    // Every frame, outside a frame. Checks file times every 0.5 s, and compiles in a process without waiting
    void update();
    bool enabled() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

} // namespace mealkit::render
