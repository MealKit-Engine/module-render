#include "mealkit/render/TextureCache.h"

#include "mealkit/core/Log.h"
#include "mealkit/core/Profile.h"
#include "mealkit/platform/Paths.h"

#include "stb_image.h"

// Files, cache folders and the dev-build rule; mips, BC7/BC5 blocks and the cache bytes are mkbake's (mkrender/Bake.h)
namespace mealkit::render {
namespace {

bool readCache(const std::string& path, CompressedTexture& out) {
    std::string file;
    mkrender::CompressedTexture t;
    if (!readTextFile(path, file) || !mkrender::readTextureFile(file, t)) return false;
    out.width = t.width;
    out.height = t.height;
    out.mipLevels = t.mipLevels;
    out.format = static_cast<asset::TextureFormat>(t.format);
    out.data = std::move(t.data);
    return true;
}

} // namespace

std::string textureCacheFolder() { return projectRoot() + "/build/cache/textures"; }

bool loadCompressedTexture(const void* encoded, size_t bytes, TextureUsage usage, CompressedTexture& out, std::string& error) {
    MEALKIT_ZONE("Texture cache");
    const std::string name = mkrender::textureCacheName(encoded, bytes, static_cast<mkrender::TextureUsage>(usage));
    const std::string written = textureCacheFolder() + "/" + name;
    if (readCache(written, out) || readCache(executableDirectory() + "/cache/textures/" + name, out)) return true;

#if MEALKIT_DEBUG_UI
    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(static_cast<const stbi_uc*>(encoded), static_cast<int>(bytes), &w, &h, &channels, 4);
    if (!pixels) {
        error = std::string("cannot decode image: ") + stbi_failure_reason();
        return false;
    }
    mkrender::CompressedTexture t;
    const bool ok = mkrender::compressTexture(pixels, static_cast<u32>(w), static_cast<u32>(h), static_cast<mkrender::TextureUsage>(usage), t);
    stbi_image_free(pixels);
    if (!ok) {
        error = "texture compression is not built in (MKBAKE_BC)";
        return false;
    }
    if (!createDirectories(textureCacheFolder()) || !writeTextFile(written, mkrender::textureFileBytes(t)))
        MEALKIT_WARN("Texture cache: cannot write %s", written.c_str());
    MEALKIT_INFO("Texture cache: compressed %ux%u, %u mips -> %s", t.width, t.height, t.mipLevels, name.c_str());
    out.width = t.width;
    out.height = t.height;
    out.mipLevels = t.mipLevels;
    out.format = static_cast<asset::TextureFormat>(t.format);
    out.data = std::move(t.data);
    return true;
#else
    error = "texture " + name + " is not in the texture cache (export the game again)";
    return false;
#endif
}

} // namespace mealkit::render
