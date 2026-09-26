#include "amp/log.h"

#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <unistd.h>

#include "amp/format.h"

namespace amp {

namespace {
std::mutex  g_log_mutex;
LogLevel    g_level     = LogLevel::kInfo;
bool        g_level_set = false;

LogLevel level_from_env() {
    const char * env = getenv("AMP_LOG_LEVEL");
    if (!env) {
        return LogLevel::kInfo;
    }
    if (!strcmp(env, "trace")) return LogLevel::kTrace;
    if (!strcmp(env, "debug")) return LogLevel::kDebug;
    if (!strcmp(env, "info"))  return LogLevel::kInfo;
    if (!strcmp(env, "warn"))  return LogLevel::kWarn;
    if (!strcmp(env, "error")) return LogLevel::kError;
    if (!strcmp(env, "off"))   return LogLevel::kOff;
    return LogLevel::kInfo;
}
} // namespace

LogLevel log_level() {
    if (!g_level_set) {
        g_level     = level_from_env();
        g_level_set = true;
    }
    return g_level;
}

void set_log_level(LogLevel lvl) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_level     = lvl;
    g_level_set = true;
}

const char * log_level_name(LogLevel lvl) {
    switch (lvl) {
        case LogLevel::kTrace: return "TRACE";
        case LogLevel::kDebug: return "DEBUG";
        case LogLevel::kInfo:  return "INFO";
        case LogLevel::kWarn:  return "WARN";
        case LogLevel::kError: return "ERROR";
        case LogLevel::kOff:   return "OFF";
    }
    return "?";
}

void log_line(LogLevel lvl, const char * file, int line, const std::string & msg) {
    const char * base = strrchr(file, '/');
    base             = base ? base + 1 : file;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tmv;
    localtime_r(&ts.tv_sec, &tmv);
    char stamp[32];
    strftime(stamp, sizeof(stamp), "%H:%M:%S", &tmv);

    // One write() so interleaved worker threads cannot split a line.
    char    line_buf[4600];
    const int n = snprintf(line_buf, sizeof(line_buf), "%s.%03ld %-5s amp/%s:%d: %s\n", stamp,
                           (long) (ts.tv_nsec / 1000000), log_level_name(lvl), base, line,
                           msg.c_str());
    if (n <= 0) {
        return;
    }
    const ssize_t ignored = write(STDERR_FILENO, line_buf, (size_t) std::min<int>(n, (int) sizeof(line_buf) - 1));
    (void) ignored;
}

} // namespace amp
