// Byte-size helpers and small numeric utilities used across modules.
#pragma once

#include <cstdint>
#include <string>

namespace amp {

inline constexpr uint64_t kKiB = 1024ull;
inline constexpr uint64_t kMiB = 1024ull * kKiB;
inline constexpr uint64_t kGiB = 1024ull * kMiB;
inline constexpr uint64_t kTiB = 1024ull * kGiB;

std::string human_bytes(uint64_t bytes);
std::string human_rate(double bytes_per_second);
std::string human_count(uint64_t n);
std::string human_time_us(uint64_t us);

// parse "4096", "4k", "6GiB", "6141MiB", "1.5 GB" -> bytes. Returns false on garbage.
bool parse_bytes(const std::string & text, uint64_t * out);
bool parse_int(const std::string & text, int64_t * out);

inline uint64_t align_up(uint64_t v, uint64_t a) { return a ? ((v + a - 1) / a) * a : v; }
inline uint64_t align_down(uint64_t v, uint64_t a) { return a ? (v / a) * a : v; }

inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

} // namespace amp
