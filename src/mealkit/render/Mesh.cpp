#include "mealkit/render/Mesh.h"

namespace mealkit::render {

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
        mesh.vertices.push_back({c - u - v, n});
        mesh.vertices.push_back({c - u + v, n});
        mesh.vertices.push_back({c + u + v, n});
        mesh.vertices.push_back({c + u - v, n});
        const u32 face[6] = {0, 1, 2, 0, 2, 3};
        for (u32 i : face) mesh.indices.push_back(base + i);
    }
    return mesh;
}

} // namespace mealkit::render
