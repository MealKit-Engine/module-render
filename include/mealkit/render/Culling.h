#pragma once

#include "mealkit/core/Math.h"

#include <algorithm>
#include <cmath>

// Frustum culling (CPU, bounding spheres). The renderer extracts planes from viewProj every frame, and DrawList::submit checks each instance.
// Affects only the view and is unrelated to the simulation (view-side float math, outside determinism rules).
namespace mealkit::render {

// Bounding sphere in mesh local space (from vertices when the mesh is made)
struct BoundingSphere {
    Vec3 center;
    f32 radius = 0.0f;
};

// Center of the vertex box + distance to the farthest vertex (not a minimal sphere, but computed in one pass)
inline BoundingSphere boundingSphere(const Vec3* points, size_t count) {
    if (count == 0) return {};
    Vec3 lo = points[0], hi = points[0];
    for (size_t i = 1; i < count; ++i) {
        lo = {std::min(lo.x, points[i].x), std::min(lo.y, points[i].y), std::min(lo.z, points[i].z)};
        hi = {std::max(hi.x, points[i].x), std::max(hi.y, points[i].y), std::max(hi.z, points[i].z)};
    }
    BoundingSphere s;
    s.center = (lo + hi) * 0.5f;
    f32 r2 = 0.0f;
    for (size_t i = 0; i < count; ++i) r2 = std::max(r2, lengthSq(points[i] - s.center));
    s.radius = std::sqrt(r2);
    return s;
}

// 6 planes (positive inside, unit-length normals). Planes without a normal (infinitely far planes) are skipped
struct Frustum {
    f32 planes[6][4] = {};
    u32 count = 0;

    // Row vector convention (clip = v * viewProj), clip space -w<=x,y<=w, 0<=z<=w (both DX12 and reversed-Z)
    static Frustum fromViewProj(const Mat4& vp) {
        Frustum f;
        const auto col = [&](int c, int row) { return vp.m[row][c]; };
        const f32 sign[6][2] = {{1, 0}, {-1, 0}, {1, 1}, {-1, 1}, {0, 2}, {-1, 2}}; // w ± x, w ± y, z, w - z
        for (const auto& s : sign) {
            f32 p[4];
            for (int row = 0; row < 4; ++row) {
                const f32 axis = col(static_cast<int>(s[1]), row);
                p[row] = s[0] == 0 ? axis : col(3, row) + s[0] * axis;
            }
            const f32 len = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            if (len < 1e-8f) continue;
            for (int k = 0; k < 4; ++k) f.planes[f.count][k] = p[k] / len;
            ++f.count;
        }
        return f;
    }

    // A world-space sphere overlaps the frustum (counts as visible when straddling the boundary)
    bool sphereVisible(Vec3 c, f32 r) const {
        for (u32 i = 0; i < count; ++i) {
            const f32* p = planes[i];
            if (p[0] * c.x + p[1] * c.y + p[2] * c.z + p[3] < -r) return false;
        }
        return true;
    }
};

} // namespace mealkit::render
