#pragma once

// Forward declarations for headers that only hold render pointers or references. Most render types are mkrender's (MkRender),
// so a plain `namespace mealkit::render { class DrawList; }` would declare a different class: include this instead.
namespace mkrender {
class DrawList;
struct FrameParams;
struct PointLight;
struct Viewport;
} // namespace mkrender

namespace mealkit::render {
using mkrender::DrawList;
using mkrender::FrameParams;
using mkrender::PointLight;
using mkrender::Viewport;
class Renderer; // the engine glue class (render/Renderer.h)
} // namespace mealkit::render
