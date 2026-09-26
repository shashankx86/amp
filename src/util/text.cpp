#include "amp/util/text.h"

namespace amp {

namespace {
void trim_left(std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        s.clear();
    } else if (b > 0) {
        s.erase(0, b);
    }
}

void trim_right(std::string & s) {
    const size_t e = s.find_last_not_of(" \t\r\n");
    if (e == std::string::npos) {
        s.clear();
    } else {
        s.resize(e + 1);
    }
}
} // namespace

std::string normalize_reasoning(const std::string & reasoning, const std::string & start_tag,
                                const std::string & end_tag) {
    std::string s = reasoning;
    trim_left(s);
    trim_right(s);

    if (!start_tag.empty() && s.compare(0, start_tag.size(), start_tag) == 0) {
        s.erase(0, start_tag.size());
        trim_left(s);
    }
    if (!end_tag.empty() && s.size() >= end_tag.size() &&
        s.compare(s.size() - end_tag.size(), end_tag.size(), end_tag) == 0) {
        s.resize(s.size() - end_tag.size());
        trim_right(s);
    }
    return s;
}

} // namespace amp
