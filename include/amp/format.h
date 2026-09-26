// Minimal printf-style string formatting, header + one small .cpp.
// Deliberately tiny: amp only needs %s-ish substitution for log lines and messages.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace amp {

// ---- stringify helpers (so callers never deal with varargs) ----
inline std::string to_str(const std::string & s) { return s; }
inline std::string to_str(const char * s) { return s ? std::string(s) : std::string("(null)"); }
inline std::string to_str(char * s) { return to_str((const char *) s); }
inline std::string to_str(char c) { return std::string(1, c); }
inline std::string to_str(bool b) { return b ? "true" : "false"; }
template <class T>
inline std::string to_str(T v) {
    if constexpr (std::is_enum_v<T>) {
        return std::to_string(static_cast<int64_t>(v));
    } else if constexpr (std::is_integral_v<T>) {
        return std::to_string(v);
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::to_string(v);
    } else {
        return std::string("<unprintable>");
    }
}

namespace detail {
std::string vformat(const char * fmt, const std::vector<std::string> & args);
}

template <class... Args>
std::string format(const char * fmt, Args &&... args) {
    if constexpr (sizeof...(Args) == 0) {
        return std::string(fmt);
    } else {
        std::vector<std::string> v;
        v.reserve(sizeof...(Args));
        (v.push_back(to_str(std::forward<Args>(args))), ...);
        return detail::vformat(fmt, v);
    }
}

} // namespace amp
