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
constexpr f64 kCheckSeconds = 0.5;         // 파일 시각을 보는 간격
const char* const kStages[2][2] = {{"VSMain", "vs_6_6"}, {"PSMain", "ps_6_6"}}; // CMakeLists의 mealkit_shader와 같다
} // namespace

struct ShaderReloader::Impl {
    struct Program {
        std::string file;
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

    std::deque<size_t> queue; // 다시 컴파일할 프로그램
    ChildProcess process;
    std::string output;
    size_t job = 0;
    u32 stage = 0; // 0 = VS, 1 = PS
    bool busy = false;

    std::string source(const std::string& file) const { return sourceDir + "/" + file; }
    std::string binary(size_t program, u32 s) const { return outDir + "/" + programs[program].file + "." + kStages[s][0] + ".bin"; }

    void startStage() {
        // ChildProcess가 cmd.exe /s /c "..."로 감싼다
        const std::string line = "\"" + dxc + "\" -nologo -T " + kStages[stage][1] + " -E " + kStages[stage][0] + " " +
                                 MEALKIT_SHADER_HOT_FLAGS + " -Fo \"" + binary(job, stage) + "\" \"" + source(programs[job].file) + "\"";
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

void ShaderReloader::add(const char* file, Apply apply) {
    if (!m->enabled) return;
    Impl::Program p;
    p.file = file;
    p.apply = std::move(apply);
    p.time = fileWriteTime(m->source(file));
    m->programs.push_back(std::move(p));
}

void ShaderReloader::update() {
    if (!m->enabled) return;
    if (m->busy) {
        if (m->process.poll(m->output)) return; // 아직 컴파일 중
        m->busy = false;
        if (m->process.exitCode() != 0) { // 오류: 로그에 남기고 이전 셰이더를 계속 쓴다
            MEALKIT_ERROR("Shader reload: %s (%s) failed, keeping the old shader\n%s", m->programs[m->job].file.c_str(),
                          kStages[m->stage][0], m->output.c_str());
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
    const bool all = include != m->includeTime; // 공통 헤더가 바뀌면 모두
    m->includeTime = include;
    for (size_t i = 0; i < m->programs.size(); ++i) {
        const unsigned long long t = fileWriteTime(m->source(m->programs[i].file));
        if (t != m->programs[i].time || all) m->enqueue(i);
        m->programs[i].time = t;
    }
}

#else // 출시 빌드: 아무것도 하지 않는다

struct ShaderReloader::Impl {};
ShaderReloader::ShaderReloader() = default;
ShaderReloader::~ShaderReloader() = default;
bool ShaderReloader::enabled() const { return false; }
void ShaderReloader::add(const char*, Apply) {}
void ShaderReloader::update() {}

#endif

} // namespace mealkit::render
