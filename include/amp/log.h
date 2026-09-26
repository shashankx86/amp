// Tiny leveled logger. No dependencies, thread-safe, writes to stderr.
// Level from AMP_LOG_LEVEL=trace|debug|info|warn|error (default: info).
//
// Call sites are stream style:  AMP_INFO("warmed ", n, " ranges in ", secs, " s")
//
// This is deliberate. The logger used to be printf-style with __attribute__((format(printf))), and
// every call site in the tree was *not* using a format string - so every multi-argument line printed
// only its first literal and silently dropped the numbers. Stream style removes the format-string
// hazard entirely (a stray % in model output cannot corrupt a log line) and makes the call sites
// read the way they were written.
#pragma once

#include <sstream>
#include <string>

namespace amp {

enum class LogLevel : int { kTrace = 0, kDebug = 1, kInfo = 2, kWarn = 3, kError = 4, kOff = 5 };

LogLevel     log_level();
void         set_log_level(LogLevel lvl);
const char * log_level_name(LogLevel lvl);

// Writes one line. Exposed for tests and for callers that already have a formatted string.
void log_line(LogLevel lvl, const char * file, int line, const std::string & msg);

// Accumulates one log line, flushes on destruction.
class LogLine {
public:
    LogLine(LogLevel lvl, const char * file, int line) : lvl_(lvl), file_(file), line_(line) {}

    ~LogLine() { log_line(lvl_, file_, line_, os_.str()); }

    LogLine(const LogLine &) = delete;
    LogLine & operator=(const LogLine &) = delete;

    template <class T>
    LogLine & operator<<(const T & value) {
        os_ << value;
        return *this;
    }

private:
    std::ostringstream os_;
    LogLevel           lvl_;
    const char *       file_;
    int                line_;
};

// The macro cannot do the streaming: `LogLine(...) << __VA_ARGS__` puts one `<<` in front of the whole
// argument list, so the commas turn into the comma *operator* and only the first argument is ever
// streamed. That bug silently truncated every multi-argument log line in the tree. A variadic
// function template with a fold expression is the fix.
template <class... Args>
void log_stream(LogLevel lvl, const char * file, int line, const Args &... args) {
    if (lvl < log_level()) {
        return;
    }
    LogLine line_out(lvl, file, line);
    (line_out << ... << args);
}

} // namespace amp

#define AMP_LOG(lvl, ...) \
    ::amp::log_stream((lvl), __FILE__, __LINE__, __VA_ARGS__)

#define AMP_TRACE(...) AMP_LOG(::amp::LogLevel::kTrace, __VA_ARGS__)
#define AMP_DEBUG(...) AMP_LOG(::amp::LogLevel::kDebug, __VA_ARGS__)
#define AMP_INFO(...)  AMP_LOG(::amp::LogLevel::kInfo, __VA_ARGS__)
#define AMP_WARN(...)  AMP_LOG(::amp::LogLevel::kWarn, __VA_ARGS__)
#define AMP_ERROR(...) AMP_LOG(::amp::LogLevel::kError, __VA_ARGS__)
