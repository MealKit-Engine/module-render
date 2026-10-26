#pragma once

#include "mealkit/render/Renderer.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mealkit::render {

// 한 프레임에 그릴 인스턴스를 메시별로 모은다. submit()하면 메시마다 드로우 콜 1회.
// submit은 renderer.beginFrame 뒤에 부르고, 그 프레임 프러스텀 밖의 인스턴스(메시 바운딩 구 기준)는 GPU에 올리지 않는다.
// 버킷 메모리는 프레임 사이에 재사용된다.
class DrawList {
public:
    void clear() {
        for (auto& bucket : m_buckets) bucket.clear();
    }

    void add(MeshId mesh, const InstanceData& instance) {
        if (mesh >= m_buckets.size()) m_buckets.resize(mesh + 1);
        m_buckets[mesh].push_back(instance);
    }

    void submit(Renderer& renderer) const {
        renderer.drawShadows(*this); // 그림자 맵 먼저 (본 화면을 묶기 전에)
        const Frustum& frustum = renderer.frustum();
        for (size_t mesh = 0; mesh < m_buckets.size(); ++mesh) {
            const auto& bucket = m_buckets[mesh];
            if (bucket.empty()) continue;
            const BoundingSphere b = renderer.meshBounds(static_cast<MeshId>(mesh));
            m_visible.clear();
            for (const InstanceData& inst : bucket) {
                const Vec3 s = inst.scale;
                const f32 scale = std::max({std::abs(s.x), std::abs(s.y), std::abs(s.z)});
                const Vec3 c = inst.position + rotate(inst.rotation, Vec3{b.center.x * s.x, b.center.y * s.y, b.center.z * s.z});
                if (frustum.sphereVisible(c, b.radius * scale)) m_visible.push_back(inst);
            }
            renderer.addCulled(static_cast<u32>(bucket.size() - m_visible.size()));
            if (!m_visible.empty())
                renderer.drawInstances(static_cast<MeshId>(mesh), m_visible.data(), static_cast<u32>(m_visible.size()));
        }
    }

    // 메시 번호 = 칸 번호 (그림자 패스가 읽는다)
    const std::vector<std::vector<InstanceData>>& buckets() const { return m_buckets; }

private:
    std::vector<std::vector<InstanceData>> m_buckets;
    mutable std::vector<InstanceData> m_visible; // submit의 작업 버퍼 (프레임 사이에 재사용)
};

} // namespace mealkit::render
