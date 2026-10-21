#pragma once

#include "mealkit/core/Math.h"

#include <vector>

namespace mealkit::render {

// 36바이트 (docs/DECISIONS.md 2026-10-21 "정점 형식"): 텍스처 좌표는 half 두 개, 머티리얼은 렌더러 머티리얼 표의 번호
struct Vertex {
    Vec3 position;
    Vec3 normal;
    u32 color = 0xFFFFFFFF; // RGBA8 sRGB. 인스턴스 색·머티리얼 기본 색과 곱해진다 (glTF COLOR_0)
    u32 uv = 0;             // TEXCOORD_0 (half u | half v << 16, packHalf2)
    u16 material = 0;       // Renderer::createMaterial 번호 (0 = 기본 흰색)
    u16 reserved = 0;
};
static_assert(sizeof(Vertex) == 36);

// float 두 개 → half 두 개 (반올림, 범위 밖은 무한대)
u32 packHalf2(f32 u, f32 v);

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<u32> indices; // 정점이 65536개 미만이면 GPU에는 16비트로 올라간다
};

// 원점 중심, 한 변 1인 상자. 면마다 정점을 따로 둬서 법선이 각지게 나온다.
MeshData makeBox();

} // namespace mealkit::render
