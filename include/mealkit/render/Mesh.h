#pragma once

#include "aster/core/Math.h"

#include <vector>

namespace aster::render {

struct Vertex {
    Vec3 position;
    Vec3 normal;
    u32 color = 0xFFFFFFFF; // RGBA8 sRGB. 인스턴스 색과 곱해진다 (glTF 재질 색)
};
static_assert(sizeof(Vertex) == 28);

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<u32> indices; // 정점이 65536개 미만이면 GPU에는 16비트로 올라간다
};

// 원점 중심, 한 변 1인 상자. 면마다 정점을 따로 둬서 법선이 각지게 나온다.
MeshData makeBox();

} // namespace aster::render
