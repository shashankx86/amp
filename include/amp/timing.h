// Monotonic stopwatch + scoped timers for the benchmark paths.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace amp {

class Stopwatch {
public:
    Stopwatch() { reset(); }
    void reset() { t0_ = std::chrono::steady_clock::now(); }

    double   elapsed_us() const { return us_since(t0_); }
    double   elapsed_ms() const { return elapsed_us() / 1e3; }
    double   elapsed_s() const { return elapsed_us() / 1e6; }
    uint64_t elapsed_us_u() const { return (uint64_t) elapsed_us(); }

    static double us_since(std::chrono::steady_clock::time_point t0) {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(now - t0).count();
    }

private:
    std::chrono::steady_clock::time_point t0_;
};

// Reads /proc/meminfo. Used to observe page-cache residency, which is the single
// most important signal in this whole project.
struct MemInfo {
    uint64_t mem_total_bytes = 0;
    uint64_t mem_available_bytes = 0;
    uint64_t cached_bytes = 0;       // Cached + SReclaimable-ish; Cached is the page cache
    uint64_t dirty_bytes = 0;

    // Rough estimate of how much of the page cache is "free for us" right now.
    uint64_t cache_headroom() const {
        const uint64_t used_by_us = mem_total_bytes - mem_available_bytes;
        return used_by_us < cached_bytes ? cached_bytes - used_by_us : 0;
    }
};

MemInfo read_meminfo();

} // namespace amp
