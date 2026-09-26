// Tiny leveled logger. No dependencies, thread-safe, writes to stderr.
// Level from AMP_LOG_LEVEL=trace|debug|info|warn|error (default: info).
#pragma once

#include <cstdio>
#include <mutex>
#include <string>

namespace amp {

enum class LogLevel : int { kTrace = 0, kDebug = 1, kInfo = 2, kWarn = 3, kError = 4, kOff = 5 };

LogLevel     log_level();
void         set_log_level(LogLevel lvl);
const char * log_level_name(LogLevel lvl);

void log_write(LogLevel lvl, const char * file, int line, const char * fmt, ...)
    __attribute__((format(printf, 4, 5)));

#define AMP_LOG(lvl, ...)                                                                     \
    do {                                                                                      \
        if ((lvl) >= ::amp::log_level()) {                                                    \
            ::amp::log_write((lvl), __FILE__, __LINE__, __VA_ARGS__);                          \
        }                                                                                     \
    } while (0)

#define AMP_TRACE(...) AMP_LOG(::amp::LogLevel::kTrace, __VA_ARGS__)
#define AMP_DEBUG(...) AMP_LOG(::amp::LogLevel::kDebug, __VA_ARGS__)
#define AMP_INFO(...)  AMP_LOG(::amp::LogLevel::kInfo,  __VA_ARGS__)
#define AMP_WARN(...)  AMP_LOG(::amp::LogLevel::kWarn,  __VA_ARGS__)
#define AMP_ERROR(...) AMP_LOG(::amp::LogLevel::kError, __VA_ARGS__)

} // namespace amp
