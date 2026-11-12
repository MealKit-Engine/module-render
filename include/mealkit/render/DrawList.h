#pragma once

#include "mealkit/render/Renderer.h"

#include <mkrender/DrawList.h>

// The per-frame draw list is mkrender's (MkRender): instances per mesh, skinned meshes, sprites and particles, culled on the job workers
// through render::Renderer (RendererDesc::parallelFor) and drawn by submit
namespace mealkit::render {

using namespace ::mkrender;

} // namespace mealkit::render
