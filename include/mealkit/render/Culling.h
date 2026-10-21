#pragma once

#include "mealkit/core/Math.h"

#include <algorithm>
#include <cmath>

// 프러스텀 컬링 (CPU, 바운딩 구). 렌더러가 프레임마다 viewProj에서 평면을 뽑고, DrawList::submit이 인스턴스마다 본다.
// 화면에만 영향이 있고 시뮬레이션과 무관하다 (화면 쪽 float 수학, 결정론 규칙 밖).
namespace mealkit::render {

// 메시 로컬 공간의 바운딩 구 (메시를 만들 때 정점에서)
struct BoundingSphere {
    Vec3 center;
    f32 radius = 0.0f;
};

// 정점 상자의 가운데 + 가장 먼 정점까지의 거리 (최소 구는 아니지만 한 번에 구한다)
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

// 평면 6개 (안쪽이 양수, 법선 길이 1). 법선이 없는 평면(무한히 먼 면)은 건너뛴다
struct Frustum {
    f32 planes[6][4] = {};
    u32 count = 0;

    // 행 벡터 규약(clip = v * viewProj), 클립 공간 -w<=x,y<=w, 0<=z<=w (DX12·reversed-Z 모두)
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

    // 월드 공간 구가 프러스텀과 겹친다 (경계에 걸치면 보이는 쪽)
    bool sphereVisible(Vec3 c, f32 r) const {
        for (u32 i = 0; i < count; ++i) {
            const f32* p = planes[i];
            if (p[0] * c.x + p[1] * c.y + p[2] * c.z + p[3] < -r) return false;
        }
        return true;
    }
};

} // namespace mealkit::render
