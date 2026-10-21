#include "mealkit/render/Mesh.h"

#include <cstring>

namespace mealkit::render {

namespace {
u16 toHalf(f32 f) {
    u32 x;
    std::memcpy(&x, &f, 4);
    const u32 sign = (x >> 16) & 0x8000u;
    const i32 exp = static_cast<i32>((x >> 23) & 0xFF) - 127 + 15;
    u32 mant = x & 0x7FFFFFu;
    if (exp <= 0) { // 아주 작은 값: 비정규 half 또는 0
        if (exp < -10) return static_cast<u16>(sign);
        mant |= 0x800000u;
        const u32 shift = static_cast<u32>(14 - exp);
        return static_cast<u16>(sign | ((mant + (1u << (shift - 1))) >> shift));
    }
    if (exp >= 31) return static_cast<u16>(sign | 0x7C00u); // 무한대 (NaN도)
    const u32 h = sign | (static_cast<u32>(exp) << 10) | (mant >> 13);
    return static_cast<u16>(h + ((mant >> 12) & 1u)); // 반올림 (올림이 지수로 넘어가도 맞다)
}
} // namespace

u32 packHalf2(f32 u, f32 v) { return static_cast<u32>(toHalf(u)) | (static_cast<u32>(toHalf(v)) << 16); }

MeshData makeBox() {
    MeshData mesh;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const Vec3 normals[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (const Vec3& n : normals) {
        Vec3 u = n.x != 0.0f ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
        Vec3 v = cross(n, u);
        // D3D 기본값은 시계 방향이 앞면이다. 아래 정점 순서에서 앞면 조건은 cross(v, u)가 n과 같은 방향일 때다.
        if (dot(cross(v, u), n) < 0.0f) {
            const Vec3 t = u;
            u = v;
            v = t;
        }
        const Vec3 c = n * 0.5f;
        u *= 0.5f;
        v *= 0.5f;

        const u32 base = static_cast<u32>(mesh.vertices.size());
        mesh.vertices.push_back({c - u - v, n, 0xFFFFFFFF, packHalf2(0, 0)});
        mesh.vertices.push_back({c - u + v, n, 0xFFFFFFFF, packHalf2(0, 1)});
        mesh.vertices.push_back({c + u + v, n, 0xFFFFFFFF, packHalf2(1, 1)});
        mesh.vertices.push_back({c + u - v, n, 0xFFFFFFFF, packHalf2(1, 0)});
        const u32 face[6] = {0, 1, 2, 0, 2, 3};
        for (u32 i : face) mesh.indices.push_back(base + i);
    }
    return mesh;
}

} // namespace mealkit::render
