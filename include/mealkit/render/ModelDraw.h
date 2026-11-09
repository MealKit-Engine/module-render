#pragma once

#include "mealkit/asset/ModelCache.h"
#include "mealkit/render/DrawList.h"
#include "mealkit/render/Renderer.h"
#include "mealkit/scene/Scene.h"

#include <string>
#include <vector>

// The view side of asset::ModelCache: uploads to the renderer and drawing models, terrain and scatter layers
// (docs/ENGINE_MODULES.md "Modules": asset holds CPU data only and never includes render).
namespace mealkit::render {

// asset::ModelGpu on a renderer: models.setGpu(&uploader) once after Renderer::init (headless sets none)
class ModelUploader final : public asset::ModelGpu {
public:
    explicit ModelUploader(Renderer& renderer) : m_renderer(renderer) {}
    MeshId createMesh(const MeshData& mesh) override;
    void replaceMesh(MeshId id, const MeshData& mesh) override;
    u32 createMaterial(const MaterialData& material) override;
    u32 createTexture(const asset::CompressedTexture& texture) override;
    asset::TextureCompressor textureCompressor() const override;
    u32 createEnvironment(const std::string& file, std::string& error) override;
    u32 createProbe(const std::string& file, std::string& error) override;
    u32 createLut(const std::string& file, std::string& error) override;

private:
    Renderer& m_renderer;
};

// Adds one model to the list: static models via add, skinned models via addSkinned with a palette looping anim ("<clip> [seconds offset]", the scene's kAnimField)
// at seconds (rest pose if no or empty clip). mesh is the ModelCache::pickLod result (lods[0] for skinned models)
void addModel(DrawList& out, const asset::Model& model, MeshId mesh, const InstanceData& instance, const std::string& anim = {},
              f64 seconds = 0.0);

// Terrain chunks with LOD by distance (instance at the box center, culled per chunk by the draw list). Builds stale meshes first (outside a frame)
void addTerrain(DrawList& out, asset::ModelCache& models, asset::Terrain& terrain, Vec3 center, Vec3 size, u32 color, u32 flags, Vec3 eye);

// A scatter object's instances (fields ScatterFields): whole cells beyond drawDistance are skipped, the rest go per instance through the draw list
void addScatter(DrawList& out, asset::ModelCache& models, const scene::Entity& layer, Vec3 eye, u32 flags = 0);

// Scene environment post values → FrameParams::post (LUT file indexed via ModelCache::lut)
PostParams postParams(asset::ModelCache& models, const scene::PostFields& fields);

// Scene reflection probe objects (scene::kTypeProbe, baked ones only) → FrameParams::probes. Positions in world coordinates (both hierarchical and flattened scenes).
// Probes over ModelCache::probeLimit (project setting probeMaxCount) and kMaxReflectionProbes are dropped
void probes(asset::ModelCache& models, const scene::Scene& scene, std::vector<ReflectionProbe>& out);

} // namespace mealkit::render
