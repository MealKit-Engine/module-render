#pragma once

#include "mealkit/rhi/Rhi.h"

#include <functional>
#include <memory>
#include <string>

// 셰이더 핫 리로드 (엔진 개발용, docs/ROADMAP.md 2단계 "셰이더 핫 리로드"). 개발 빌드(MEALKIT_SHADER_HOT_RELOAD,
// 디버그 UI가 있는 빌드)에서 engine/shaders의 .hlsl·Bindless.hlsli가 바뀌면 빌드와 같은 DXC·같은 옵션으로 다시 컴파일하고
// 파이프라인을 바꿔 끼운다. 컴파일 오류는 로그에 남기고 이전 셰이더를 계속 쓴다. 출시 빌드(build.bat Export)는 박아 넣은
// 바이트코드만 쓰고 이것은 아무것도 하지 않는다. 엔진 폴더나 DXC가 없는 PC(복사한 개발 exe)에서도 꺼진다.
namespace mealkit::render {

class ShaderReloader {
public:
    // 바이트코드를 받아 파이프라인을 다시 만든다. 실패하면 false (이전 것을 계속 쓴다)
    using Apply = std::function<bool(const rhi::ShaderCode& vs, const rhi::ShaderCode& ps)>;

    ShaderReloader();
    ~ShaderReloader();
    ShaderReloader(const ShaderReloader&) = delete;
    ShaderReloader& operator=(const ShaderReloader&) = delete;

    // engine/shaders/<file>의 VSMain + PSMain
    void add(const char* file, Apply apply);
    // 매 프레임, 프레임 밖에서. 0.5초마다 파일 시각을 보고, 컴파일은 프로세스로 돌려 기다리지 않는다
    void update();
    bool enabled() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

} // namespace mealkit::render
