#include "amp/util/json.h"

#include "amp/bytes.h"
#include "amp/format.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace amp {
namespace json {

namespace {

const Value kNullValue{};

void append_escaped(std::string & out, const std::string & s) {
    out.push_back('"');
    for (const unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (int) c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
        }
    }
    out.push_back('"');
}

void append_double(std::string & out, double d) {
    if (std::isnan(d) || std::isinf(d)) {
        out += "null";
        return;
    }
    char buf[40];
    // 17 significant digits round-trips an IEEE double exactly; trim what JSON does not need.
    snprintf(buf, sizeof(buf), "%.17g", d);
    out += buf;
}

class Parser {
public:
    explicit Parser(const std::string & t) : t_(t) {}

    Result<Value> run() {
        skip_ws();
        Result<Value> v = parse_value(0);
        if (!v.ok()) {
            return v;
        }
        skip_ws();
        if (i_ != t_.size()) {
            return Status::Errorf("trailing characters at offset %zu", i_);
        }
        return v;
    }

private:
    Status fail(const char * what) const {
        return Status::Errorf("json: %s at offset %zu", what, i_);
    }

    void skip_ws() {
        while (i_ < t_.size()) {
            const char c = t_[i_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                i_++;
            } else {
                break;
            }
        }
    }

    bool consume(char c) {
        if (i_ < t_.size() && t_[i_] == c) {
            i_++;
            return true;
        }
        return false;
    }

    bool literal(const char * lit) {
        const size_t n = strlen(lit);
        if (t_.compare(i_, n, lit) == 0) {
            i_ += n;
            return true;
        }
        return false;
    }

    Result<Value> parse_value(int depth) {
        if (depth > 200) {
            return fail("nesting too deep");
        }
        skip_ws();
        if (i_ >= t_.size()) {
            return fail("unexpected end of input");
        }
        const char c = t_[i_];
        switch (c) {
            case 'n': return literal("null") ? Result<Value>(Value()) : Result<Value>(fail("bad null"));
            case 't': return literal("true") ? Result<Value>(Value(true)) : Result<Value>(fail("bad true"));
            case 'f': return literal("false") ? Result<Value>(Value(false)) : Result<Value>(fail("bad false"));
            case '"': {
                Result<std::string> s = parse_string();
                if (!s.ok()) {
                    return s.status();
                }
                return Value(*s);
            }
            case '[': return parse_array(depth);
            case '{': return parse_object(depth);
            default:  return parse_number();
        }
    }

    Result<std::string> parse_string() {
        if (!consume('"')) {
            return fail("expected '\"'");
        }
        std::string out;
        while (i_ < t_.size()) {
            const unsigned char c = (unsigned char) t_[i_++];
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out.push_back((char) c);
                continue;
            }
            if (i_ >= t_.size()) {
                break;
            }
            const char e = t_[i_++];
            switch (e) {
                case '"':  out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'u': {
                    if (i_ + 4 > t_.size()) {
                        return fail("truncated \\u escape");
                    }
                    const unsigned cp = (unsigned) strtoul(t_.substr(i_, 4).c_str(), nullptr, 16);
                    i_ += 4;
                    // Encode as UTF-8. Surrogate pairs are passed through as replacement chars:
                    // llama.cpp token ids and prose do not need full pair handling here.
                    if (cp < 0x80) {
                        out.push_back((char) cp);
                    } else if (cp < 0x800) {
                        out.push_back((char) (0xC0 | (cp >> 6)));
                        out.push_back((char) (0x80 | (cp & 0x3F)));
                    } else {
                        out.push_back((char) (0xE0 | (cp >> 12)));
                        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back((char) (0x80 | (cp & 0x3F)));
                    }
                    break;
                }
                default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }

    Result<Value> parse_number() {
        const size_t start = i_;
        if (i_ < t_.size() && (t_[i_] == '-' || t_[i_] == '+')) {
            i_++;
        }
        bool is_double = false;
        while (i_ < t_.size()) {
            const char c = t_[i_];
            if (c >= '0' && c <= '9') {
                i_++;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
                is_double = true;
                i_++;
            } else {
                break;
            }
        }
        if (i_ == start) {
            return fail("expected a value");
        }
        const std::string num = t_.substr(start, i_ - start);
        if (!is_double) {
            errno = 0;
            char *     end = nullptr;
            const long long v = strtoll(num.c_str(), &end, 10);
            if (errno == 0 && end && *end == '\0') {
                return Value((int64_t) v);
            }
        }
        return Value(strtod(num.c_str(), nullptr));
    }

    Result<Value> parse_array(int depth) {
        consume('[');
        Array a;
        skip_ws();
        if (consume(']')) {
            return Value(std::move(a));
        }
        while (true) {
            Result<Value> v = parse_value(depth + 1);
            if (!v.ok()) {
                return v;
            }
            a.push_back(std::move(*v));
            skip_ws();
            if (consume(',')) {
                continue;
            }
            if (consume(']')) {
                break;
            }
            return fail("expected ',' or ']'");
        }
        return Value(std::move(a));
    }

    Result<Value> parse_object(int depth) {
        consume('{');
        Object o;
        skip_ws();
        if (consume('}')) {
            return Value(std::move(o));
        }
        while (true) {
            skip_ws();
            Result<std::string> k = parse_string();
            if (!k.ok()) {
                return k.status();
            }
            skip_ws();
            if (!consume(':')) {
                return fail("expected ':'");
            }
            Result<Value> v = parse_value(depth + 1);
            if (!v.ok()) {
                return v;
            }
            o[*k] = std::move(*v);
            skip_ws();
            if (consume(',')) {
                continue;
            }
            if (consume('}')) {
                break;
            }
            return fail("expected ',' or '}'");
        }
        return Value(std::move(o));
    }

    const std::string & t_;
    size_t             i_ = 0;
};

} // namespace

// ---------------------------------------------------------------- Value

bool Value::as_bool(bool def) const {
    switch (type_) {
        case Type::Bool:   return b_;
        case Type::Int:    return i_ != 0;
        case Type::Double: return d_ != 0.0;
        case Type::String: return s_ == "true" || s_ == "1";
        default:           return def;
    }
}

int64_t Value::as_int(int64_t def) const {
    switch (type_) {
        case Type::Int:    return i_;
        case Type::Double: return (int64_t) d_;
        case Type::Bool:   return b_ ? 1 : 0;
        case Type::String: return strtoll(s_.c_str(), nullptr, 10);
        default:           return def;
    }
}

double Value::as_double(double def) const {
    switch (type_) {
        case Type::Double: return d_;
        case Type::Int:    return (double) i_;
        case Type::Bool:   return b_ ? 1.0 : 0.0;
        case Type::String: return strtod(s_.c_str(), nullptr);
        default:           return def;
    }
}

std::string Value::as_string(const std::string & def) const {
    return type_ == Type::String ? s_ : def;
}

const Array & Value::as_array() const {
    static const Array empty;
    return type_ == Type::Array ? a_ : empty;
}

Array & Value::as_array() {
    if (type_ != Type::Array) {
        type_ = Type::Array;
        a_.clear();
    }
    return a_;
}

const Object & Value::as_object() const {
    static const Object empty;
    return type_ == Type::Object ? o_ : empty;
}

Object & Value::as_object() {
    if (type_ != Type::Object) {
        type_ = Type::Object;
        o_.clear();
    }
    return o_;
}

const Value * Value::get(const std::string & key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }
    const auto it = o_.find(key);
    return it == o_.end() ? nullptr : &it->second;
}

bool Value::has(const std::string & key) const { return get(key) != nullptr; }

Value & Value::operator[](const std::string & key) {
    as_object();
    return o_[key];
}

size_t Value::size() const {
    if (type_ == Type::Array) {
        return a_.size();
    }
    if (type_ == Type::Object) {
        return o_.size();
    }
    if (type_ == Type::String) {
        return s_.size();
    }
    return 0;
}

const Value & Value::at(size_t i) const {
    if (type_ == Type::Array && i < a_.size()) {
        return a_[i];
    }
    return kNullValue;
}

std::string Value::dump(int indent) const {
    std::string out;
    dump_to(out, indent, 0);
    return out;
}

void Value::dump_to(std::string & out, int indent, int depth) const {
    const bool pretty = indent >= 0;
    const std::string nl = pretty ? "\n" : "";
    const std::string pad = pretty ? std::string((size_t) (indent * (depth + 1)), ' ') : "";
    const std::string pad_end = pretty ? std::string((size_t) (indent * depth), ' ') : "";

    switch (type_) {
        case Type::Null:   out += "null"; break;
        case Type::Bool:   out += b_ ? "true" : "false"; break;
        case Type::Int:    out += std::to_string(i_); break;
        case Type::Double: append_double(out, d_); break;
        case Type::String: append_escaped(out, s_); break;
        case Type::Array: {
            if (a_.empty()) {
                out += "[]";
                break;
            }
            out += "[" + nl;
            for (size_t i = 0; i < a_.size(); i++) {
                out += pad;
                a_[i].dump_to(out, indent, depth + 1);
                if (i + 1 < a_.size()) {
                    out += ",";
                }
                out += nl;
            }
            out += pad_end + "]";
            break;
        }
        case Type::Object: {
            if (o_.empty()) {
                out += "{}";
                break;
            }
            out += "{" + nl;
            size_t i = 0;
            for (const auto & kv : o_) {
                out += pad;
                append_escaped(out, kv.first);
                out += pretty ? ": " : ":";
                kv.second.dump_to(out, indent, depth + 1);
                if (++i < o_.size()) {
                    out += ",";
                }
                out += nl;
            }
            out += pad_end + "}";
            break;
        }
    }
}

Result<Value> Value::parse(const std::string & text) { return Parser(text).run(); }

} // namespace json
} // namespace amp
