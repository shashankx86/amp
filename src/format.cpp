#include "amp/format.h"

#include <cstdio>
#include <cstring>

namespace amp {
namespace detail {

// Substitutes %s-style conversions with already-stringified arguments. Supports
// %s %d %i %u %f %%. Width/precision are honoured for %f only (truncation), which is
// all amp's log lines need.
std::string vformat(const char * fmt, const std::vector<std::string> & args) {
    std::string out;
    out.reserve(strlen(fmt) + 32 * args.size());

    size_t next_arg = 0;
    for (const char * p = fmt; *p; ++p) {
        if (*p != '%') {
            out.push_back(*p);
            continue;
        }
        // collect the spec: %[-+ #0]*[0-9]*(\.[0-9]+)?[hlL]*[conversion]
        const char * spec_start = p;
        ++p;
        while (*p && strchr("-+ #0", *p)) {
            ++p;
        }
        while (*p >= '0' && *p <= '9') {
            ++p;
        }
        int precision = -1;
        if (*p == '.') {
            ++p;
            precision = 0;
            while (*p >= '0' && *p <= '9') {
                precision = precision * 10 + (*p - '0');
                ++p;
            }
        }
        while (*p && strchr("hlLqjzt", *p)) {
            ++p;
        }
        const char conv = *p;
        if (conv == '\0') {
            out.append(spec_start, p);
            break;
        }
        if (conv == '%') {
            out.push_back('%');
            continue;
        }
        const std::string & arg = (next_arg < args.size()) ? args[next_arg] : std::string("?");
        ++next_arg;
        if (conv == 'f' || conv == 'g' || conv == 'e') {
            char buf[64];
            if (precision >= 0) {
                snprintf(buf, sizeof(buf), "%.*f", precision, atof(arg.c_str()));
            } else {
                snprintf(buf, sizeof(buf), "%s", arg.c_str());
            }
            out += buf;
        } else {
            out += arg;
        }
    }
    return out;
}

} // namespace detail
} // namespace amp
