// Minimal JSON: parse, inspect, serialise.
//
// Hand-written rather than borrowed from the vendored tree on purpose. The vendored json.hpp is an
// internal header of another project; coupling the API layer to it would make llama.cpp's internal
// refactors our problem. This is the subset a JSON API needs, and it is unit tested.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "amp/status.h"

namespace amp {
namespace json {

class Value;
using Array  = std::vector<Value>;
using Object = std::map<std::string, Value>;

enum class Type { Null, Bool, Int, Double, String, Array, Object };

class Value {
public:
    // NOTE: every converting constructor is explicit on purpose. With implicit ones,
    // `const Value & v = some_ptr;` silently binds a temporary Value(true) instead of
    // dereferencing, because a raw pointer converts to bool. That is not hypothetical - it shipped,
    // and the test suite caught it. Explicit also means a missed `*` becomes a compile error.
    Value() = default;
    explicit Value(std::nullptr_t) : type_(Type::Null) {}
    explicit Value(bool b) : type_(Type::Bool), b_(b) {}
    explicit Value(int i) : type_(Type::Int), i_(i) {}
    explicit Value(int64_t i) : type_(Type::Int), i_(i) {}
    explicit Value(uint64_t i) : type_(Type::Int), i_((int64_t) i) {}
    explicit Value(double d) : type_(Type::Double), d_(d) {}
    explicit Value(float d) : type_(Type::Double), d_((double) d) {}
    explicit Value(const char * s) : type_(Type::String), s_(s ? s : "") {}
    explicit Value(std::string s) : type_(Type::String), s_(std::move(s)) {}
    explicit Value(Array a) : type_(Type::Array), a_(std::move(a)) {}
    explicit Value(Object o) : type_(Type::Object), o_(std::move(o)) {}

    static Value array() { return Value(Array{}); }
    static Value object() { return Value(Object{}); }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_int() const { return type_ == Type::Int; }
    bool is_double() const { return type_ == Type::Double; }
    bool is_number() const { return type_ == Type::Int || type_ == Type::Double; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool               as_bool(bool def = false) const;
    int64_t            as_int(int64_t def = 0) const;
    double             as_double(double def = 0.0) const;
    std::string        as_string(const std::string & def = "") const;
    const Array &      as_array() const;
    Array &            as_array();
    const Object &     as_object() const;
    Object &           as_object();

    // object access: get() returns nullptr for missing keys
    const Value *      get(const std::string & key) const;
    bool               has(const std::string & key) const;
    Value &            operator[](const std::string & key);       // creates objects as needed

    // Convenience setters. Templated rather than taking Value directly so callers can write
    // set("n", 3) / push("x"), while a pointer argument is still a compile error rather than a
    // silent Value(true): there is no Value(const Value *) constructor to select.
    template <class T>
    void set(const std::string & key, T && v) {
        as_object()[key] = Value(std::forward<T>(v));
    }

    // array access
    size_t             size() const;
    const Value &      at(size_t i) const;
    template <class T>
    void push(T && v) {
        as_array().push_back(Value(std::forward<T>(v)));
    }

    std::string dump(int indent = -1) const;

    static Result<Value> parse(const std::string & text);

private:
    void dump_to(std::string & out, int indent, int depth) const;

    Type        type_;
    bool        b_    = false;
    int64_t     i_    = 0;
    double      d_    = 0.0;
    std::string s_;
    Array       a_;
    Object      o_;
};

} // namespace json
} // namespace amp
