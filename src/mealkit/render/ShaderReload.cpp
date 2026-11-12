#include "mealkit/render/ShaderReload.h"

#if MEALKIT_SHADER_HOT_RELOAD
#include "mealkit/core/Clock.h"
#include "mealkit/core/Log.h"
#include "mealkit/platform/Paths.h"
#include "mealkit/platform/Process.h"

#include <deque>
#include <vector>
#endif

namespace mealkit::render {

#if MEALKIT_SHADER_HOT_RELOAD

namespace {
constexpr f64 kCheckSeconds = 0.5;         // interval for checking file times
const char* const kProfiles[2] = {"vs_6_6", "ps_6_6"}; // same as mkrender_shader in MkRender's CMakeLists.txt
} // namespace

struct ShaderReloader::Impl {
    struct Program {
        std::string file;
        std::string entries[2]; // VS, PS
        std::string defines;
        Apply apply;
        unsigned long long time = 0;
    };
    std::vector<Program> programs;
    std::string sourceDir = MEALKIT_SHADER_SOURCE_DIR;
    std::string outDir = MEALKIT_SHADER_HOT_DIR;
    std::string dxc = MEALKIT_SHADER_DXC_PATH;
    bool enabled = false;
    unsigned long long includeTime = 0;
    f64 nextCheck = 0.0;

    std::deque<size_t> queue; // programs to recompile
    ChildProcess process;
    std::string output;
    size_t job = 0;
    u32 stage = 0; // 0 = VS, 1 = PS
    bool busy = false;

    std::string source(const std::string& file) const { return sourceDir + "/" + file; }
    // A different file name per variant (lighting model) of the same file
    std::string binary(size_t program, u32 s) const {
        return outDir + "/" + programs[program].file + "." + std::to_string(program) + "." + programs[program].entries[s] + ".bin";
    }

    void startStage() {
        // ChildProcess wraps it in cmd.exe /s /c "..."
        const Program& p = programs[job];
        const std::string define = p.defines.empty() ? std::string() : " -D " + p.defines;
        // SPIR-V (Vulkan builds) renames entry points to VSMain/PSMain (same as mkrender_shader in MkRender's CMakeLists.txt)
        const std::string flags = MEALKIT_SHADER_HOT_FLAGS;
        const std::string entryName =
            flags.find("-spirv") != std::string::npos ? std::string(" -fspv-entrypoint-name=") + (stage == 0 ? "VSMain" : "PSMain") : "";
        const std::string line = "\"" + dxc + "\" -nologo -T " + kProfiles[stage] + " -E " + p.entries[stage] + " " +
                                 flags + entryName + define + " -Fo \"" + binary(job, stage) + "\" \"" + source(p.file) + "\"";
        output.clear();
        busy = process.start(line, sourceDir);
        if (!busy) MEALKIT_WARN("Shader reload: cannot run %s", dxc.c_str());
    }

    void finishProgram() {
        std::string vs, ps;
        if (!readTextFile(binary(job, 0), vs) || !readTextFile(binary(job, 1), ps)) {
            MEALKIT_WARN("Shader reload: cannot read the compiled %s", programs[job].file.c_str());
            return;
        }
        if (programs[job].apply({vs.data(), vs.size()}, {ps.data(), ps.size()}))
            MEALKIT_INFO("Shader reloaded: %s", programs[job].file.c_str());
        else MEALKIT_WARN("Shader reload: pipeline for %s not replaced, keeping the old one", programs[job].file.c_str());
    }

    void enqueue(size_t i) {
        for (size_t q : queue)
            if (q == i) return;
        queue.push_back(i);
    }
};

ShaderReloader::ShaderReloader() : m(std::make_unique<Impl>()) {
    m->enabled = directoryExists(m->sourceDir) && fileExists(m->dxc) && createDirectories(m->outDir);
    if (m->enabled) m->includeTime = fileWriteTime(m->source("Bindless.hlsli"));
}

ShaderReloader::~ShaderReloader() = default;

bool ShaderReloader::enabled() const { return m->enabled; }

void ShaderReloader::add(const char* file, Apply apply, const char* vsEntry, const char* psEntry, const char* defines) {
    if (!m->enabled) return;
    Impl::Program p;
    p.file = file;
    p.entries[0] = vsEntry;
    p.entries[1] = psEntry;
    p.defines = defines;
    p.apply = std::move(apply);
    p.time = fileWriteTime(m->source(file));
    m->programs.push_back(std::move(p));
}

void ShaderReloader::update() {
    if (!m->enabled) return;
    if (m->busy) {
        if (m->process.poll(m->output)) return; // still compiling
        m->busy = false;
        if (m->process.exitCode() != 0) { // error: log it and keep using the previous shader
            MEALKIT_ERROR("Shader reload: %s (%s) failed, keeping the old shader\n%s", m->programs[m->job].file.c_str(),
                          m->programs[m->job].entries[m->stage].c_str(), m->output.c_str());
        } else if (m->stage == 0) {
            m->stage = 1;
            m->startStage();
            return;
        } else {
            m->finishProgram();
        }
    }
    if (!m->queue.empty()) {
        m->job = m->queue.front();
        m->queue.pop_front();
        m->stage = 0;
        m->startStage();
        return;
    }

    const f64 now = nowSeconds();
    if (now < m->nextCheck) return;
    m->nextCheck = now + kCheckSeconds;
    const unsigned long long include = fileWriteTime(m->source("Bindless.hlsli"));
    const bool all = include != m->includeTime; // all when the common header changes
    m->includeTime = include;
    for (size_t i = 0; i < m->programs.size(); ++i) {
        const unsigned long long t = fileWriteTime(m->source(m->programs[i].file));
        if (t != m->programs[i].time || all) m->enqueue(i);
        m->programs[i].time = t;
    }
}

#else // shipping builds: do nothing

struct ShaderReloader::Impl {};
ShaderReloader::ShaderReloader() = default;
ShaderReloader::~ShaderReloader() = default;
bool ShaderReloader::enabled() const { return false; }
void ShaderReloader::add(const char*, Apply, const char*, const char*, const char*) {}
void ShaderReloader::update() {}

#endif

} // namespace mealkit::render
