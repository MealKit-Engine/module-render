#pragma once

#include "mealkit/jobs/Jobs.h"
#include "mealkit/render/Renderer.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mealkit::render {

// Culling work buffers, reused between frames: visible[mesh] = that bucket's instances inside the frustum (in bucket order)
struct CullScratch {
    std::vector<std::vector<InstanceData>> visible;
    std::vector<u8> inside;   // one flag per instance, all buckets back to back
    std::vector<size_t> first; // first flag of each bucket (+ total at the end)
};

// Frustum culling of the first meshCount buckets on the job workers (mealkit/jobs/Jobs.h).
// The flags are computed in parallel over all instances (balanced even when one bucket holds most of them, e.g. scatter grass),
// then each bucket is compacted in order, so the result is the same as a serial loop.
template <class BoundsOf>
void cullInstances(const std::vector<std::vector<InstanceData>>& buckets, size_t meshCount, const Frustum& frustum, BoundsOf&& boundsOf,
                   CullScratch& out) {
    out.visible.resize(std::max(out.visible.size(), meshCount));
    out.first.resize(meshCount + 1);
    size_t total = 0;
    for (size_t mesh = 0; mesh < meshCount; ++mesh) {
        out.first[mesh] = total;
        total += buckets[mesh].size();
    }
    out.first[meshCount] = total;
    out.inside.resize(total);
    constexpr u32 kChunk = 512; // const-ok: instances per job (smaller isn't worth a job)
    jobs::parallelFor(static_cast<u32>(total), kChunk, [&](u32 begin, u32 end) {
        size_t mesh = static_cast<size_t>(std::upper_bound(out.first.begin(), out.first.end(), begin) - out.first.begin()) - 1;
        BoundingSphere b = boundsOf(mesh);
        for (u32 i = begin; i < end; ++i) {
            while (i >= out.first[mesh + 1]) b = boundsOf(++mesh);
            const InstanceData& inst = buckets[mesh][i - out.first[mesh]];
            const Vec3 s = inst.scale;
            const f32 scale = std::max({std::abs(s.x), std::abs(s.y), std::abs(s.z)});
            const Vec3 c = inst.position + rotate(inst.rotation, Vec3{b.center.x * s.x, b.center.y * s.y, b.center.z * s.z});
            out.inside[i] = frustum.sphereVisible(c, b.radius * scale) ? 1 : 0;
        }
    });
    for (size_t mesh = 0; mesh < meshCount; ++mesh) {
        std::vector<InstanceData>& visible = out.visible[mesh];
        visible.clear();
        const auto& bucket = buckets[mesh];
        for (size_t k = 0; k < bucket.size(); ++k)
            if (out.inside[out.first[mesh] + k]) visible.push_back(bucket[k]);
    }
}

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
        cullInstances(m_buckets, m_buckets.size(), frustum, [&](size_t mesh) { return renderer.meshBounds(static_cast<MeshId>(mesh)); }, m_cull);
        for (size_t mesh = 0; mesh < m_buckets.size(); ++mesh) {
            const auto& visible = m_cull.visible[mesh];
            renderer.addCulled(static_cast<u32>(m_buckets[mesh].size() - visible.size()));
            if (!visible.empty())
                renderer.drawInstances(static_cast<MeshId>(mesh), visible.data(), static_cast<u32>(visible.size()));
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
    mutable CullScratch m_cull; // submit's culling buffers (reused between frames)
};

} // namespace mealkit::render
