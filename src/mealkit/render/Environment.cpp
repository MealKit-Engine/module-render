#include "mealkit/render/Environment.h"
#include "mealkit/asset/Cooked.h"

#include "mealkit/core/Log.h"
#include "mealkit/core/Profile.h"
#include "mealkit/platform/Paths.h"
#include "mealkit/render/Renderer.h"

#include <algorithm>
#include <string_view>

#if MEALKIT_DEBUG_UI
#include "stb_image.h"
#endif

// Files, cache folders and the dev-build rule; the baking is mkbake's (mkrender/Bake.h)
namespace mealkit::render {

bool saveProbe(const std::string& path, const ProbeMap& probe) {
    const size_t slash = path.find_last_of("/\\");
    if (slash != std::string::npos) createDirectories(path.substr(0, slash));
    return writeTextFile(path, mkrender::probeFileBytes(probe));
}

bool loadProbe(const std::string& path, ProbeMap& out, std::string& error) {
    std::string file;
    if (!readTextFile(path, file)) {
        error = "file not found (bake the probe in the editor)";
        return false;
    }
    return mkrender::readProbeFile(file, out, error);
}

// Files with that extension anywhere under the kind folder kind/ (any subfolders), paths relative to assets
static std::vector<std::string> listKindFiles(const std::string& assetsRoot, const char* kind, const char* extension) {
    std::vector<std::string> out;
    std::vector<std::string> folders{kind};
    while (!folders.empty()) {
        const std::string folder = folders.back();
        folders.pop_back();
        for (const std::string& f : listFiles(assetsRoot + "/" + folder, extension)) out.push_back(folder + "/" + f);
        for (const std::string& d : listDirectories(assetsRoot + "/" + folder)) folders.push_back(folder + "/" + d);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::string> listHdris(const std::string& assetsRoot) { return listKindFiles(assetsRoot, "textures", ".hdr"); }
std::vector<std::string> listLuts(const std::string& assetsRoot) { return listKindFiles(assetsRoot, "lut", ".cube"); }

bool parseCubeLut(const std::string& text, std::vector<f32>& out, std::string& error) { return mkrender::parseCubeLut(text, kLutSize, out, error); }

bool loadCubeLut(const std::string& path, std::vector<f32>& out, std::string& error) {
    std::string text;
    if (!readTextFile(path, text)) return error = "file not found", false;
    return parseCubeLut(text, out, error);
}

bool saveNeutralCubeLut(const std::string& path) { return writeTextFile(path, mkrender::neutralCubeLut(kLutSize)); }

std::string environmentCacheFolder() { return projectRoot() + "/build/cache/environments"; }

std::string environmentCacheFile(const std::string& hdrPath) {
    std::string file;
    if (!readTextFile(hdrPath, file)) return {};
    const std::string written = environmentCacheFolder() + "/" + mkrender::environmentCacheName(file);
    return fileExists(written) ? written : std::string();
}

static bool readCache(const std::string& path, EnvironmentMap& out) {
    std::string file;
    return readTextFile(path, file) && mkrender::readEnvironmentFile(file, out);
}

bool loadEnvironment(const std::string& hdrPath, EnvironmentMap& out, std::string& error) {
    MEALKIT_ZONE("Environment cache");
    const std::string_view cooked = asset::kCookedEnvironmentSuffix;
    if (hdrPath.size() > cooked.size() && hdrPath.compare(hdrPath.size() - cooked.size(), cooked.size(), cooked) == 0) {
        if (readCache(hdrPath, out)) return true; // cooked into the pack: the cache file as is
        error = "cooked environment of another engine version (cook the game again)";
        return false;
    }
    std::string file;
    if (!readTextFile(hdrPath, file)) {
        error = "file not found";
        return false;
    }
    const std::string name = mkrender::environmentCacheName(file);
    const std::string written = environmentCacheFolder() + "/" + name;
    if (readCache(written, out) || readCache(executableDirectory() + "/cache/environments/" + name, out)) return true;

#if MEALKIT_DEBUG_UI
    int w = 0, h = 0, channels = 0;
    f32* rgb = stbi_loadf_from_memory(reinterpret_cast<const stbi_uc*>(file.data()), static_cast<int>(file.size()), &w, &h, &channels, 3);
    if (!rgb) {
        error = std::string("cannot decode HDR image: ") + stbi_failure_reason();
        return false;
    }
    {
        MEALKIT_ZONE("Environment bake");
        bakeEnvironment(rgb, static_cast<u32>(w), static_cast<u32>(h), out);
    }
    stbi_image_free(rgb);
    if (!createDirectories(environmentCacheFolder()) || !writeTextFile(written, mkrender::environmentFileBytes(out)))
        MEALKIT_WARN("Environment cache: cannot write %s", written.c_str());
    MEALKIT_INFO("Environment cache: baked %dx%d HDRI (sky %u, reflection %u, %u mips) -> %s", w, h, out.skySize,
                 out.reflectSize, out.reflectMips, name.c_str());
    return true;
#else
    error = "environment " + name + " is not in the environment cache (export the game again)";
    return false;
#endif
}

} // namespace mealkit::render
