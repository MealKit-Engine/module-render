#pragma once

// MkRender host types (mkrender/Config.h): the engine's own math and mesh types are mkrender's, so engine and game code
// passes them to the renderer as they are. The engine sets MKRENDER_USER_CONFIG to this header for the library and every engine target.
#include "mealkit/asset/MeshData.h"
#include "mealkit/core/Math.h"

// mkrender::Image stays the library's own (platform/Image.h is not a core header): render::Renderer converts toon ramps and captures
#define MKRENDER_HOST_MATH 1
#define MKRENDER_HOST_MESH 1

namespace mkrender {

using mealkit::cross;
using mealkit::dot;
using mealkit::length;
using mealkit::lengthSq;
using mealkit::lerp;
using mealkit::lookAtLH;
using mealkit::Mat4;
using mealkit::normalize;
using mealkit::perspectiveReversedZ;
using mealkit::Quat;
using mealkit::rotate;
using mealkit::Vec2;
using mealkit::Vec3;

using mealkit::asset::BoneMatrix;
using mealkit::asset::kMaterialDissolve;
using mealkit::asset::kMaterialOutline;
using mealkit::asset::kMaterialRim;
using mealkit::asset::kMaterialWind;
using mealkit::asset::kNoMesh;
using mealkit::asset::kNoTexture;
using mealkit::asset::MaterialData;
using mealkit::asset::MeshData;
using mealkit::asset::MeshId;
using mealkit::asset::SkinWeights;
using mealkit::asset::Vertex;

} // namespace mkrender
