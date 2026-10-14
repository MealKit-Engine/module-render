#pragma once

#include "aster/render/Renderer.h"

#include <vector>

namespace aster::render {

// 한 프레임에 그릴 인스턴스를 메시별로 모은다. submit()하면 메시마다 드로우 콜 1회.
// 버킷 메모리는 프레임 사이에 재사용된다.
class DrawList {
public:
    void clear() {
        for (auto& bucket : m_buckets) bucket.clear();
    }

    void add(MeshId mesh, const InstanceData& instance) {
        if (mesh >= m_buckets.size()) m_buckets.resize(mesh + 1);
        m_buckets[mesh].push_back(instance);
    }

    void submit(Renderer& renderer) const {
        for (size_t mesh = 0; mesh < m_buckets.size(); ++mesh) {
            const auto& bucket = m_buckets[mesh];
            if (!bucket.empty())
                renderer.drawInstances(static_cast<MeshId>(mesh), bucket.data(), static_cast<u32>(bucket.size()));
        }
    }

private:
    std::vector<std::vector<InstanceData>> m_buckets;
};

} // namespace aster::render
