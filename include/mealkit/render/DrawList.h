#pragma once

#include "mealkit/render/Renderer.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mealkit::render {

// Gathers instances to draw in one frame per mesh. submit() issues one draw call per mesh.
// Call submit after renderer.beginFrame; instances outside that frame's frustum (by mesh bounding sphere) aren't uploaded to the GPU.
// Bucket memory is reused between frames.
class DrawList {
public:
    void clear() {
        for (auto& bucket : m_buckets) bucket.clear();
        m_skinned.clear();
        m_bones.clear();
        m_particles.clear();
    }

    // Particles (feedback::Player::addParticles): drawn all at once after opaque 3D
    void addParticles(const Particle* particles, size_t count) { m_particles.insert(m_particles.end(), particles, particles + count); }
    const std::vector<Particle>& particles() const { return m_particles; }

    // Skinned meshes (characters): the palette is copied (asset::skinPalette). Drawn one at a time, not batched per mesh
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
    // Culling sphere for skinned meshes: the bind pose bounds enlarged by an animation margin
    static constexpr f32 kSkinnedBoundsScale = 1.5f;

    void add(MeshId mesh, const InstanceData& instance) {
        if (mesh >= m_buckets.size()) m_buckets.resize(mesh + 1);
        m_buckets[mesh].push_back(instance);
    }

    void submit(Renderer& renderer) const {
        renderer.drawShadows(*this); // shadow maps first (before binding the main view)
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
        if (!m_particles.empty()) renderer.drawParticles(m_particles.data(), static_cast<u32>(m_particles.size()));
    }

    // Mesh index = bucket index (read by the shadow pass)
    const std::vector<std::vector<InstanceData>>& buckets() const { return m_buckets; }

private:
    std::vector<std::vector<InstanceData>> m_buckets;
    std::vector<SkinnedItem> m_skinned;
    std::vector<BoneMatrix> m_bones;
    std::vector<Particle> m_particles;
    mutable std::vector<InstanceData> m_visible; // submit's work buffer (reused between frames)
};

} // namespace mealkit::render
