#pragma once

#include "mealkit/asset/MeshData.h"

// CPU mesh data lives in the asset module (asset/MeshData.h); these names keep render:: code and games unchanged
namespace mealkit::render {

using asset::BoneMatrix;
using asset::kSkinnedVertexStride;
using asset::makeBox;
using asset::MeshData;
using asset::packHalf2;
using asset::SkinWeights;
using asset::Vertex;

} // namespace mealkit::render
