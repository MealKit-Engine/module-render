#include "mealkit/render/ModelDraw.h"

#include "mealkit/asset/Animation.h"
#include "mealkit/asset/Scatter.h"
#include "mealkit/asset/Terrain.h"
#include "mealkit/render/Environment.h"
#include "mealkit/render/TextureCache.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace mealkit::render {

MeshId ModelUploader::createMesh(const MeshData& mesh) { return m_renderer.createMesh(mesh); }

void ModelUploader::replaceMesh(MeshId id, const MeshData& mesh) { m_renderer.replaceMesh(id, mesh); }

u32 ModelUploader::createMaterial(const MaterialData& material) { return m_renderer.createMaterial(material); }

// Textures: compressed by the import (read from the texture cache or compressed, render/TextureCache.cpp), uploaded here
u32 ModelUploader::createTexture(const asset::CompressedTexture& t) {
    return m_renderer.createTexture({t.width, t.height, t.mipLevels, textureFormat(t.format), t.data.data()});
}

asset::TextureCompressor ModelUploader::textureCompressor() const { return &loadCompressedTexture; }

// Sky HDRI: read from the cache or bake and upload (render/Environment.cpp)
u32 ModelUploader::createEnvironment(const std::string& file, std::string& error) {
    EnvironmentMap e;
    if (!loadEnvironment(file, e, error)) return 0;
    EnvironmentData d;
    std::memcpy(d.sh, e.sh, sizeof(d.sh));
    d.skySize = e.skySize;
    d.sky = e.sky.data();
    d.reflectSize = e.reflectSize;
    d.reflectMips = e.reflectMips;
    d.reflect = e.reflect.data();
    return m_renderer.createEnvironment(d);
}

// Reflection probes: the baked files from assets/probes/
u32 ModelUploader::createProbe(const std::string& file, std::string& error) {
    ProbeMap probe;
    if (!loadProbe(file, probe, error)) return 0;
    return m_renderer.createProbe(probe.size, probe.mips, probe.reflect.data());
}

// Color grading LUT: read the .cube and upload
u32 ModelUploader::createLut(const std::string& file, std::string& error) {
    std::vector<f32> rgb;
    if (!loadCubeLut(file, rgb, error)) return 0;
    return m_renderer.createLut(rgb.data());
}

void addModel(DrawList& out, const asset::Model& model, MeshId mesh, const InstanceData& instance, const std::string& anim, f64 seconds) {
    if (mesh == kNoMesh) return; // pickLod: below the cull size
    if (!model.skinned()) {
        out.add(mesh, instance);
        return;
    }
    thread_local std::vector<asset::JointPose> pose;
    thread_local std::vector<BoneMatrix> palette;
    const size_t space = anim.find(' ');
    const asset::AnimationClip* clip = anim.empty() ? nullptr : asset::findClip(model.clips, anim.substr(0, space));
    if (clip) {
        const f64 offset = space == std::string::npos ? 0.0 : std::atof(anim.c_str() + space + 1);
        asset::samplePose(model.skeleton, *clip, static_cast<f32>(std::fmod(seconds + offset, 4096.0)), true, pose);
    } else {
        asset::restPose(model.skeleton, pose);
    }
    asset::skinPalette(model.skeleton, pose, palette);
    out.addSkinned(model.lods[0], instance, palette.data(), static_cast<u32>(palette.size()));
}

void addTerrain(DrawList& out, asset::ModelCache& models, asset::Terrain& t, Vec3 center, Vec3 size, u32 color, u32 flags, Vec3 eye) {
    if (!models.buildTerrainMeshes(t, size)) return;
    const u32 chunks = t.chunks;
    const f32 cellX = size.x / static_cast<f32>(t.map.size - 1), cellZ = size.z / static_cast<f32>(t.map.size - 1);
    const f32 chunkWidth = std::max(cellX, cellZ) * static_cast<f32>(asset::kTerrainChunkCells);
    for (u32 cz = 0; cz < chunks; ++cz)
        for (u32 cx = 0; cx < chunks; ++cx) {
            const u32 chunk = cz * chunks + cx;
            const Vec3 middle = center - size * 0.5f +
                                Vec3{(static_cast<f32>(cx) + 0.5f) * cellX * asset::kTerrainChunkCells, size.y * 0.5f,
                                     (static_cast<f32>(cz) + 0.5f) * cellZ * asset::kTerrainChunkCells};
            const u32 lod = asset::terrainLod(length(middle - eye), chunkWidth);
            InstanceData inst;
            inst.position = center;
            inst.color = color;
            inst.flags = flags;
            out.add(t.meshes[chunk * asset::kTerrainLods + lod], inst);
        }
}

void addScatter(DrawList& out, asset::ModelCache& models, const scene::Entity& layer, Vec3 eye, u32 flags) {
    const scene::ScatterFields f = scene::readObject<scene::ScatterFields>(layer);
    if (f.file.empty()) return;
    asset::Scatter& s = models.scatter(f.file).scatter;
    if (s.items.empty()) return;
    if (!s.grouped) asset::groupScatter(s);
    const asset::Model* model = f.model.empty() ? nullptr : models.get(f.model);
    const MeshId box = model ? 0u : models.boxMesh();
    const f32 reach = f.drawDistance + asset::kScatterCellSize * 0.75f; // cell half diagonal
    for (const asset::ScatterCell& cell : s.cells) {
        const Vec3 d = cell.center - eye;
        if (d.x * d.x + d.y * d.y + d.z * d.z > reach * reach) continue;
        for (u32 k = 0; k < cell.count; ++k) {
            const asset::ScatterInstance& item = s.items[s.order[cell.first + k]];
            InstanceData inst;
            inst.rotation = quatFromAxisAngle({0.0f, 1.0f, 0.0f}, item.yaw);
            inst.color = model ? 0xFFFFFFFFu : layer.color;
            inst.flags = flags;
            if (model) {
                inst.position = item.position;
                inst.scale = {item.scale, item.scale, item.scale};
                out.add(models.pickLod(*model, eye, inst.position, inst.scale), inst);
            } else { // placeholder: a small box standing on the ground
                inst.scale = Vec3{0.3f, 0.6f, 0.3f} * item.scale;
                inst.position = item.position + Vec3{0.0f, inst.scale.y * 0.5f, 0.0f};
                out.add(box, inst);
            }
        }
    }
}

PostParams postParams(asset::ModelCache& models, const scene::PostFields& f) {
    PostParams p;
    p.toneMapping = f.toneMapping;
    p.punchy = f.punchy;
    p.exposure = f.exposure;
    p.lut = models.lut(f.lut);
    p.bloomIntensity = f.bloomIntensity;
    p.bloomThreshold = f.bloomThreshold;
    p.bloomTint = f.bloomTint;
    p.vignette = f.vignette;
    p.vignetteSmoothness = f.vignetteSmoothness;
    p.grain = f.grain;
    p.chromaticAberration = f.chromaticAberration;
    return p;
}

void probes(asset::ModelCache& models, const scene::Scene& scene, std::vector<ReflectionProbe>& out) {
    out.clear();
    const u32 limit = std::min(models.probeLimit(), kMaxReflectionProbes);
    for (const scene::Entity& e : scene.entities) {
        if (!e.is(scene::kTypeProbe) || out.size() >= limit) continue;
        const u32 id = models.probe(scene::probeFile(e));
        const scene::Transform world = scene::worldTransform(scene, e);
        if (id) out.push_back({world.position, Vec3{std::fabs(world.scale.x), std::fabs(world.scale.y), std::fabs(world.scale.z)} * 0.5f, id});
    }
}

} // namespace mealkit::render
