#pragma once

#include "mealkit/asset/MeshData.h"
#include "mealkit/rhi/Rhi.h"
#include "mkrender/Bake.h"

#include <string>
#include <vector>

// Model texture cache (docs/ROADMAP.md stage 2 "Textures and materials", docs/DECISIONS.md 2026-10-21 "Texture cache").
// PNG/JPEG inside glTF is compressed with mipmaps to BC7 (color/metal-roughness/emissive) or BC5 (normals) and saved to a file:
//   <project>/build/cache/textures/<content hash>.mktex   (outside git, regenerated if deleted)
// Compression only happens in dev builds (builds with debug UI: editor/dev exe). Shipping builds only read the compressed cache
// (Export copies it to cache/textures next to the exe). View-only, so unrelated to the simulation and content hash
// (the source images are already in the glTF file hash).
namespace mealkit::render {

using asset::CompressedTexture;
using asset::TextureUsage;

// Cooked models store the format as asset::TextureFormat: the values must stay rhi::Format's
static_assert(static_cast<u8>(asset::TextureFormat::Bc7Unorm) == static_cast<u8>(rhi::Format::Bc7Unorm) &&
              static_cast<u8>(asset::TextureFormat::Bc7Srgb) == static_cast<u8>(rhi::Format::Bc7Srgb) &&
              static_cast<u8>(asset::TextureFormat::Bc5Unorm) == static_cast<u8>(rhi::Format::Bc5Unorm));
inline rhi::Format textureFormat(asset::TextureFormat f) { return static_cast<rhi::Format>(f); }
// The compression is mkbake's: its usage values are asset::TextureUsage's (they are in the cache names)
static_assert(static_cast<u8>(TextureUsage::Color) == static_cast<u8>(mkrender::TextureUsage::Color) &&
              static_cast<u8>(TextureUsage::Linear) == static_cast<u8>(mkrender::TextureUsage::Linear) &&
              static_cast<u8>(TextureUsage::Normal) == static_cast<u8>(mkrender::TextureUsage::Normal));

// Encoded image (PNG/JPEG bytes in glTF) → read from the cache, or if missing decode, mipmap, compress, and write to cache. On failure, false and a reason
bool loadCompressedTexture(const void* encoded, size_t bytes, TextureUsage usage, CompressedTexture& out, std::string& error);

// Cache folder: written to <project>/build/cache/textures; when reading, also cache/textures next to the exe (shipping)
std::string textureCacheFolder();

// Mipmaps only without compression (for unit tests, RGBA8). With srgb, converts to linear before averaging (mkbake, mkrender/Bake.h)
using mkrender::buildMips;

} // namespace mealkit::render
