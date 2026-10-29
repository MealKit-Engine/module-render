#pragma once

#include "mealkit/core/Math.h"

#include <vector>

namespace mealkit::render {

// 36 bytes (docs/DECISIONS.md 2026-10-21 "Vertex format"): texture coordinates are two halfs, material is an index into the renderer's material table
struct Vertex {
    Vec3 position;
    Vec3 normal;
    u32 color = 0xFFFFFFFF; // RGBA8 sRGB. Multiplied with the instance color and material base color (glTF COLOR_0)
    u32 uv = 0;             // TEXCOORD_0 (half u | half v << 16, packHalf2)
    u16 material = 0;       // Renderer::createMaterial index (0 = default white)
    u16 reserved = 0;
};
static_assert(sizeof(Vertex) == 36);

// two floats → two halfs (rounded, out of range becomes infinity)
u32 packHalf2(f32 u, f32 v);

// Four joints of a skinned vertex (skeletal animation): indices u8 ×4, weights unorm8 ×4 (sum 255). On the GPU appended after Vertex for 44 bytes
struct SkinWeights {
    u32 joints = 0;
    u32 weights = 0xFFu; // first joint 1
};
static_assert(sizeof(SkinWeights) == 8);
inline constexpr u32 kSkinnedVertexStride = sizeof(Vertex) + sizeof(SkinWeights);

// One slot of the skin palette (48 bytes read by the HLSL vertex shader: three rows, p' = (dot(r0, p1), dot(r1, p1), dot(r2, p1))). Filled by asset::skinPalette
struct BoneMatrix {
    f32 rows[3][4];
};
static_assert(sizeof(BoneMatrix) == 48);

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<u32> indices; // with fewer than 65536 vertices, uploaded to the GPU as 16-bit
    std::vector<SkinWeights> skin; // empty for static meshes, one per vertex if present (skinned mesh)
};

// Box centered at the origin with side 1. Each face has its own vertices so normals come out faceted.
MeshData makeBox();

} // namespace mealkit::render
