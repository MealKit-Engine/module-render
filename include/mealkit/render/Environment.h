#pragma once

#include "mealkit/core/Math.h"
#include "mkrender/Bake.h"

#include <string>
#include <vector>

// Sky HDRI → ambient/reflections (docs/ROADMAP.md stage 2 "Sky HDRI and ambient", docs/DECISIONS.md 2026-10-26 HDRI).
// Baked from the scene environment's single `.hdr` (equirectangular Radiance) on load and saved to a file:
//   <project>/build/cache/environments/<content hash>.mkenv   (outside git, rebaked if deleted)
//   - sky cubemap (skybox, one mip)
//   - reflection cubemap 128² + mips (mip k = roughness k / (mip count − 1), GGX importance samples)
//   - 9 ambient spherical harmonic coefficients (Lambert convolution multiplied in, so shaders get the "ambient color" directly from the normal)
// Cube face order/orientation follows DX12/Vulkan rules (+X, −X, +Y, −Y, +Z, −Z, MealKit left-handed y-up), texels are RGB9E5.
// Baking happens only in dev builds (builds with debug UI): shipping builds only read the cache (Export copies it to cache/environments next to the exe).
// View-only, so unrelated to the simulation and content hash.
namespace mealkit::render {

// The baking itself is MkRender's mkbake (mkrender/Bake.h), shared with the MkRender scene editor: same cache and probe bytes
using mkrender::EnvironmentMap;
using mkrender::ProbeMap;

// hdrPath (.hdr file) → read from cache, or if missing (dev build) bake and write to cache. On failure, false and a reason.
// A path ending in .mkenv (asset/Cooked.h, the cook step) is read as a cache file directly
bool loadEnvironment(const std::string& hdrPath, EnvironmentMap& out, std::string& error);

// .hdr files under assets/textures/ (including subfolders) (paths relative to assets like "textures/hdri/x.hdr", sorted by name). Selection list for the editor [Environment] window and lookdev
std::vector<std::string> listHdris(const std::string& assetsRoot);

// Color grading LUT (.cube, 3D LUT exported by Resolve, Photoshop etc.): sRGB values resampled to kLutSize³ cells (red fastest,
// Renderer::createLut). Trilinear interpolation if sizes differ, honors DOMAIN_MIN/MAX. 1D LUTs and bad files return false and a reason
bool parseCubeLut(const std::string& text, std::vector<f32>& out, std::string& error);
bool loadCubeLut(const std::string& path, std::vector<f32>& out, std::string& error);
// .cube files under assets/lut/ (environment panel selection, kind folder lut)
std::vector<std::string> listLuts(const std::string& assetsRoot);
// Writes a neutral LUT as .cube (placed with the grading capture as a starting point in external tools)
bool saveNeutralCubeLut(const std::string& path);

// Cache folder: written to <project>/build/cache/environments; when reading, also cache/environments next to the exe (shipping)
std::string environmentCacheFolder();
// The cache file loadEnvironment wrote for an .hdr (the cook step copies it into the pack). Empty if not baked yet
std::string environmentCacheFile(const std::string& hdrPath);

// Reflection probes (docs/ROADMAP.md stage 2 "Reflection probes"): at a scene reflection_probe object's position, the editor renders the scene to six cube faces
// and reads them back (editor/ProbeBaker), and they're baked here into a reflection cube + GGX mips. Files are assets/probes/<scene>_<object id>.mkprobe (committed:
// shipping reads these files without baking). Same layout as the sky HDRI reflection cube (RGB9E5, per face starting from mip 0)
using mkrender::cubeFaceForward; // the camera of cube face i (lookAtLH(eye, eye + forward, up), 90 degrees)
using mkrender::cubeFaceUp;
using mkrender::bakeProbe; // faces: six faces (+X -X +Y -Y +Z -Z, cubeDirection orientation) x faceSize^2 x linear rgb. Size is faceSize, mips down to 4
bool saveProbe(const std::string& path, const ProbeMap& probe);
bool loadProbe(const std::string& path, ProbeMap& out, std::string& error);

// --- Baking parts (unit tests)
using mkrender::bakeEnvironment; // from an equirectangular image (rgb float, width x height), no file/cache
using mkrender::cubeDirection;
using mkrender::evalSh;
using mkrender::packRgb9e5;
using mkrender::unpackRgb9e5;

} // namespace mealkit::render
