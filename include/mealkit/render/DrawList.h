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
        m_skinned.clear();
        m_bones.clear();
    }

    // 스킨 메시 (캐릭터): 팔레트를 복사해 둔다 (asset::skinPalette). 메시마다 묶지 않고 하나씩 그린다
    void addSkinned(MeshId mesh, const InstanceData& instance, const BoneMatrix* bones, u32 boneCount) {
        m_skinned.push_back({mesh, instance, static_cast<u32>(m_bones.size()), boneCount});
        m_bones.insert(m_bones.end(), bones, bones + boneCount);
    }
    struct SkinnedItem {
        MeshId mesh;
        InstanceData instance;
        u32 firstBone, boneCount;
    };
    const std::vector<SkinnedItem>& skinned() const { return m_skinned; }
    const BoneMatrix* bones(const SkinnedItem& item) const { return m_bones.data() + item.firstBone; }
    // 스킨 메시의 컬링 구: 바인드 자세 경계를 애니메이션 여유만큼 키운다
    static constexpr f32 kSkinnedBoundsScale = 1.5f;

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
        for (const SkinnedItem& item : m_skinned) {
            const BoundingSphere b = renderer.meshBounds(item.mesh);
            const InstanceData& inst = item.instance;
            const Vec3 s = inst.scale;
            const f32 scale = std::max({std::abs(s.x), std::abs(s.y), std::abs(s.z)});
            const Vec3 c = inst.position + rotate(inst.rotation, Vec3{b.center.x * s.x, b.center.y * s.y, b.center.z * s.z});
            if (frustum.sphereVisible(c, b.radius * scale * kSkinnedBoundsScale))
                renderer.drawSkinned(item.mesh, inst, bones(item), item.boneCount);
            else
                renderer.addCulled(1);
        }
    }

    // 메시 번호 = 칸 번호 (그림자 패스가 읽는다)
    const std::vector<std::vector<InstanceData>>& buckets() const { return m_buckets; }

private:
    std::vector<std::vector<InstanceData>> m_buckets;
    std::vector<SkinnedItem> m_skinned;
    std::vector<BoneMatrix> m_bones;
    mutable std::vector<InstanceData> m_visible; // submit의 작업 버퍼 (프레임 사이에 재사용)
};

} // namespace mealkit::render
