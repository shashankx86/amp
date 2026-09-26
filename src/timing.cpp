#include "amp/timing.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace amp {

MemInfo read_meminfo() {
    MemInfo mi;
    FILE *  f = fopen("/proc/meminfo", "r");
    if (!f) {
        return mi;
    }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        unsigned long long kb = 0;
        if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) {
            mi.mem_total_bytes = kb * 1024ull;
        } else if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            mi.mem_available_bytes = kb * 1024ull;
        } else if (sscanf(line, "Cached: %llu kB", &kb) == 1) {
            mi.cached_bytes += kb * 1024ull;
        } else if (sscanf(line, "SReclaimable: %llu kB", &kb) == 1) {
            mi.cached_bytes += kb * 1024ull;  // reclaimable slab counts as usable cache
        } else if (sscanf(line, "Dirty: %llu kB", &kb) == 1) {
            mi.dirty_bytes = kb * 1024ull;
        }
    }
    fclose(f);
    return mi;
}

} // namespace amp
