// Thin C ABI over unmodified Clipper 6.4.2, matching pyclipper's defaults.
// Clipper license/provenance: ../vendor/clipper6/{LICENSE.txt,UPSTREAM.json}.
#include "clipper.hpp"
#include <cstdint>
#include <cstddef>
#include <cmath>
extern "C" std::ptrdiff_t rd_seal_offset(const std::int64_t* xy,
    std::size_t n, double delta, std::int64_t* result, std::size_t capacity) noexcept {
    try {
        if (!xy || !result || n < 3 || n > 1000000 || !std::isfinite(delta)) return -3;
        ClipperLib::Path path;
        path.reserve(n);
        for (std::size_t i = 0; i < n; ++i) path.emplace_back(xy[2*i], xy[2*i+1]);
        ClipperLib::ClipperOffset offset;
        offset.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
        ClipperLib::Paths paths;
        offset.Execute(paths, delta);
        if (paths.empty()) return 0;
        if (paths.size() != 1) return -1;
        if (paths[0].size() > capacity) return -2;
        for (std::size_t i = 0; i < paths[0].size(); ++i) {
            result[2*i] = paths[0][i].X; result[2*i+1] = paths[0][i].Y;
        }
        return static_cast<std::ptrdiff_t>(paths[0].size());
    } catch (...) { return -3; }
}
