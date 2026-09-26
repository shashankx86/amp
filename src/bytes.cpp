#include "amp/bytes.h"

#include "amp/format.h"

#include <cctype>
#include <cstdlib>
#include <cstring>

namespace amp {

std::string human_bytes(uint64_t bytes) {
    static const char * units[] = { "B", "KiB", "MiB", "GiB", "TiB" };
    double      v   = (double) bytes;
    int         idx = 0;
    while (v >= 1024.0 && idx < 4) {
        v /= 1024.0;
        idx++;
    }
    if (idx == 0) {
        return std::to_string(bytes) + " B";
    }
    return format("%.2f ", v) + units[idx];
}

std::string human_rate(double bytes_per_second) {
    return human_bytes((uint64_t) (bytes_per_second < 0 ? 0 : bytes_per_second)) + "/s";
}

std::string human_count(uint64_t n) {
    static const char * units[] = { "", "K", "M", "G", "T" };
    double      v   = (double) n;
    int         idx = 0;
    while (v >= 1000.0 && idx < 4) {
        v /= 1000.0;
        idx++;
    }
    if (idx == 0) {
        return std::to_string(n);
    }
    return format("%.2f", v) + units[idx];
}

std::string human_time_us(uint64_t us) {
    if (us < 1000) {
        return std::to_string(us) + " us";
    }
    if (us < 1000000) {
        return format("%.2f ms", us / 1e3);
    }
    return format("%.2f s", us / 1e6);
}

bool parse_int(const std::string & text, int64_t * out) {
    if (text.empty()) {
        return false;
    }
    char * end = nullptr;
    const long long v = strtoll(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    *out = (int64_t) v;
    return true;
}

bool parse_bytes(const std::string & text_in, uint64_t * out) {
    std::string text = text_in;
    while (!text.empty() && isspace((unsigned char) text.back())) {
        text.pop_back();
    }
    size_t p = 0;
    while (p < text.size() && isspace((unsigned char) text[p])) {
        p++;
    }
    char * end = nullptr;
    const double v = strtod(text.c_str() + p, &end);
    if (end == text.c_str() + p) {
        return false;
    }
    std::string suffix;
    while (*end && !isspace((unsigned char) *end)) {
        suffix.push_back((char) tolower((unsigned char) *end));
        end++;
    }
    double mult = 1.0;
    if (suffix.empty() || suffix == "b") {
        mult = 1.0;
    } else if (suffix == "k" || suffix == "kib" || suffix == "kb") {
        mult = (double) kKiB;
    } else if (suffix == "m" || suffix == "mib" || suffix == "mb") {
        mult = (double) kMiB;
    } else if (suffix == "g" || suffix == "gib" || suffix == "gb") {
        mult = (double) kGiB;
    } else if (suffix == "t" || suffix == "tib" || suffix == "tb") {
        mult = (double) kTiB;
    } else {
        return false;
    }
    *out = (uint64_t) (v * mult);
    return true;
}

} // namespace amp
