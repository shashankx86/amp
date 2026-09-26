// Result/Status: the error channel for every module boundary in amp.
//
// Rule: functions that can fail return Result<T> (or Status). No exceptions cross a
// module boundary; throwing is reserved for programming errors (assert-like checks).
#pragma once

#include <cassert>
#include <string>
#include <utility>

#include "amp/format.h"

namespace amp {

class Status {
public:
    Status() = default;

    static Status OK() { return Status(); }
    static Status Error(std::string msg) {
        Status s;
        s.ok_    = false;
        s.msg_   = std::move(msg);
        return s;
    }
    // sprintf-style convenience
    template <class... Args>
    static Status Errorf(const char * fmt, Args &&... args) {
        return Error(format(fmt, std::forward<Args>(args)...));
    }
    bool ok() const { return ok_; }
    explicit operator bool() const { return ok_; }
    const std::string & message() const { return msg_; }
    std::string        to_string() const { return ok_ ? std::string("OK") : "error: " + msg_; }

    // Prefix the message with context, keeping the error state.
    Status context(const std::string & what) const {
        if (ok_) {
            return *this;
        }
        return Error(what + ": " + msg_);
    }

private:
    bool        ok_  = true;
    std::string msg_;
};

template <class T>
class Result {
public:
    Result(T value) : value_(std::move(value)) {}          // NOLINT(google-explicit-constructor)
    Result(Status s) : status_(std::move(s)) { assert(!status_.ok()); }

    bool ok() const { return status_.ok(); }
    explicit operator bool() const { return ok(); }

    const Status & status() const { return status_; }
    std::string    message() const { return status_.message(); }

    T & operator*() { assert(ok()); return value_; }
    const T & operator*() const { assert(ok()); return value_; }
    T * operator->() { assert(ok()); return &value_; }
    const T * operator->() const { assert(ok()); return &value_; }

    T value_or(T fallback) const { return ok() ? value_ : std::move(fallback); }
    // move the payload out (for move-only types)
    T take() { return std::move(value_); }

    // Allows `return some_operation_returning_Result<T>(...)` in a Status-returning function.
    operator Status() const { return status_; }

    Status status_with_context(const std::string & what) const {
        return ok() ? Status::OK() : status_.context(what);
    }

private:
    Status status_{};
    T      value_{};
};

} // namespace amp
