#include "amp/server/http_server.h"

#include "amp/format.h"
#include "amp/log.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <thread>

namespace amp {
namespace http {

namespace {
constexpr size_t kMaxHeaderBytes = 64 * 1024;
constexpr size_t kMaxBodyBytes   = 32 * 1024 * 1024;   // 32 MiB: plenty for a 200k-token prompt

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char) tolower(c); });
    return s;
}

const char * status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default:  return "OK";
    }
}

// Read until \r\n\r\n (headers) with a size cap, then read the body per Content-Length or
// chunked framing. Returns false on clean EOF.
bool read_request(int fd, std::string & buf, Request & req) {
    size_t header_end = std::string::npos;
    // headers
    while (true) {
        header_end = buf.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            break;
        }
        if (buf.size() > kMaxHeaderBytes) {
            return false;
        }
        struct pollfd p { fd, POLLIN, 0 };
        const int    pr = poll(&p, 1, 30000);
        if (pr <= 0) {
            return false;
        }
        char tmp[8192];
        const ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            return false;
        }
        buf.append(tmp, (size_t) n);
    }

    const std::string head = buf.substr(0, header_end);
    buf.erase(0, header_end + 4);

    // request line
    size_t pos = head.find("\r\n");
    const std::string line = head.substr(0, pos);
    {
        const size_t sp1 = line.find(' ');
        const size_t sp2 = line.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos) {
            return false;
        }
        req.method = line.substr(0, sp1);
        std::string target = line.substr(sp1 + 1, sp2 - sp1 - 1);
        const size_t q = target.find('?');
        if (q == std::string::npos) {
            req.path = url_decode(target);
        } else {
            req.path = url_decode(target.substr(0, q));
            req.query = target.substr(q + 1);
        }
    }

    // headers
    while (pos != std::string::npos) {
        const size_t eol = head.find("\r\n", pos + 2);
        const std::string h = head.substr(pos + 2, (eol == std::string::npos ? head.size() : eol) - pos - 2);
        const size_t colon = h.find(':');
        if (colon != std::string::npos) {
            std::string k = to_lower(h.substr(0, colon));
            std::string v = h.substr(colon + 1);
            while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) {
                v.erase(v.begin());
            }
            while (!v.empty() && (v.back() == ' ' || v.back() == '\t' || v.back() == '\r')) {
                v.pop_back();
            }
            req.headers[k] = v;
        }
        pos = eol;
    }

    const std::string te = req.header("transfer-encoding");
    const std::string cl = req.header("content-length");
    if (te.find("chunked") != std::string::npos) {
        // Chunked request bodies are not used by any OpenAI client; read and discard framing.
        return false;
    }
    if (cl.empty()) {
        req.content_length = 0;
        return true;
    }
    req.content_length = atoi(cl.c_str());
    if (req.content_length < 0 || (size_t) req.content_length > kMaxBodyBytes) {
        req.content_length = -1;
        return false;
    }
    while (buf.size() < (size_t) req.content_length) {
        struct pollfd p { fd, POLLIN, 0 };
        if (poll(&p, 1, 30000) <= 0) {
            return false;
        }
        char tmp[65536];
        const ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            return false;
        }
        buf.append(tmp, (size_t) n);
    }
    req.body = buf.substr(0, (size_t) req.content_length);
    buf.erase(0, (size_t) req.content_length);
    return true;
}

} // namespace

std::string Request::header(const std::string & name, const std::string & def) const {
    const auto it = headers.find(to_lower(name));
    return it == headers.end() ? def : it->second;
}

bool Request::wants_stream() const {
    const std::string s = header("accept");
    return s.find("text/event-stream") != std::string::npos;
}

bool Request::is_chunked() const { return false; }

std::string url_decode(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const std::string hex = s.substr(i + 1, 2);
            out.push_back((char) strtol(hex.c_str(), nullptr, 16));
            i += 2;
        } else if (s[i] == '+') {
            out.push_back(' ');
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
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
                    out += format("\\u%04x", (int) c);
                } else {
                    out.push_back((char) c);
                }
        }
    }
    return out;
}

std::string sse_event(const std::string & data, const std::string & event) {
    std::string out;
    if (!event.empty()) {
        out += "event: " + event + "\n";
    }
    out += "data: " + data + "\n\n";
    return out;
}

// ---------------------------------------------------------------- ResponseWriter

Status ResponseWriter::send_headers(int status, const std::string & content_type, bool chunked,
                                    const std::vector<std::pair<std::string, std::string>> & extra) {
    if (headers_sent_) {
        return Status::OK();
    }
    std::string h = format("HTTP/1.1 %d %s\r\n", status, status_text(status));
    h += "Content-Type: " + content_type + "\r\n";
    h += "Server: amp\r\n";
    h += "Access-Control-Allow-Origin: *\r\n";
    h += "Access-Control-Allow-Headers: *\r\n";
    h += "Access-Control-Allow-Methods: *\r\n";
    h += "Connection: keep-alive\r\n";
    for (const auto & kv : extra) {
        h += kv.first + ": " + kv.second + "\r\n";
    }
    if (chunked) {
        h += "Transfer-Encoding: chunked\r\n";
    }
    h += "\r\n";
    const Status st = write_raw(h.data(), h.size());
    headers_sent_ = true;
    chunked_      = chunked;
    return st;
}

Status ResponseWriter::write_raw(const char * data, size_t len) {
    size_t done = 0;
    while (done < len) {
        const ssize_t n = ::send(fd_, data + done, len - done, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return Status::Errorf("send failed: %s", strerror(errno));
        }
        done += (size_t) n;
    }
    bytes_ += len;
    return Status::OK();
}

Status ResponseWriter::write(const std::string & data) {
    if (!headers_sent_) {
        return Status::Error("ResponseWriter: headers not sent");
    }
    if (chunked_) {
        char hdr[32];
        const int n = snprintf(hdr, sizeof(hdr), "%zx\r\n", data.size());
        Status st = write_raw(hdr, (size_t) n);
        if (!st.ok()) {
            return st;
        }
        st = write_raw(data.data(), data.size());
        if (!st.ok()) {
            return st;
        }
        return write_raw("\r\n", 2);
    }
    return write_raw(data.data(), data.size());
}

Status ResponseWriter::write_sse(const std::string & data) { return write(sse_event(data)); }

Status ResponseWriter::write_sse_done() { return write("data: [DONE]\n\n"); }

Status ResponseWriter::finish() {
    if (!headers_sent_) {
        return Status::Error("ResponseWriter: finish before headers");
    }
    if (chunked_) {
        return write_raw("0\r\n\r\n", 5);
    }
    return Status::OK();
}

// ---------------------------------------------------------------- Server

Server::~Server() {
    if (listen_fd_ >= 0) {
        close(listen_fd_);
    }
}

Result<std::unique_ptr<Server>> Server::listen(const std::string & host, uint16_t port, int n_threads) {
    (void) n_threads;
    auto srv = std::unique_ptr<Server>(new Server());

    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return Status::Errorf("socket: %s", strerror(errno));
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (host == "0.0.0.0" || host.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        close(fd);
        return Status::Errorf("bad host '%s'", host.c_str());
    }

    if (bind(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        const int e = errno;
        close(fd);
        return Status::Errorf("bind %s:%u failed: %s", host.c_str(), port, strerror(e));
    }
    if (::listen(fd, 64) != 0) {
        const int e = errno;
        close(fd);
        return Status::Errorf("listen failed: %s", strerror(e));
    }

    // Report the actual port when 0 was requested (useful for tests).
    if (port == 0) {
        sockaddr_in bound{};
        socklen_t   blen = sizeof(bound);
        if (getsockname(fd, (sockaddr *) &bound, &blen) == 0) {
            srv->port_ = ntohs(bound.sin_port);
        }
    } else {
        srv->port_ = port;
    }
    srv->listen_fd_ = fd;
    return srv;
}

void Server::route(const std::string & method, const std::string & path, Handler h) {
    const std::string key = method + " " + path;
    if (routes_.find(key) == routes_.end()) {
        order_.push_back(key);
    }
    routes_[key] = std::move(h);
}

void Server::stop() {
    stop_ = true;
    if (listen_fd_ >= 0) {
        shutdown(listen_fd_, SHUT_RDWR);
    }
}

Status Server::serve() {
    signal(SIGPIPE, SIG_IGN);
    std::vector<std::thread> workers;
    const int                 n = std::max(2, (int) std::thread::hardware_concurrency() / 4);
    for (int i = 0; i < n; i++) {
        workers.emplace_back([this] {
            while (!stop_) {
                struct pollfd p { listen_fd_, POLLIN, 0 };
                const int    pr = poll(&p, 1, 200);
                if (pr <= 0) {
                    continue;
                }
                const int cfd = accept(listen_fd_, nullptr, nullptr);
                if (cfd < 0) {
                    continue;
                }
                int one = 1;
                setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                const Status st = handle_connection(cfd);
                if (!st.ok()) {
                    AMP_DEBUG("amp: connection ended: ", st.message());
                }
                close(cfd);
            }
        });
    }
    for (auto & t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    return Status::OK();
}

Status Server::handle_connection(int fd) {
    std::string buf;
    int         served = 0;
    while (!stop_ && served < 1000) {
        Request req;
        if (!read_request(fd, buf, req)) {
            return Status::OK();
        }
        ResponseWriter w(fd);
        const Status    st = dispatch(req, w);
        if (!st.ok()) {
            return st;
        }
        if (!w.finish().ok()) {
            return Status::Error("finish failed");
        }
        served++;
        if (!req.keep_alive || buf.empty()) {
            break;
        }
    }
    return Status::OK();
}

Status Server::send_error(ResponseWriter & w, int status, const std::string & message) {
    if (w.headers_sent()) {
        return Status::Error("error after headers: " + message);
    }
    const std::string body =
        format("{\"error\":{\"message\":%s,\"type\":\"amp_error\",\"code\":%d}}",
               ("\"" + json_escape(message) + "\"").c_str(), status);
    (void) w.send_headers(status, "application/json", false);
    return w.write(body);
}

Status Server::dispatch(const Request & req, ResponseWriter & w) {
    auto it = routes_.find(req.method + " " + req.path);
    if (it == routes_.end()) {
        // Distinguish "wrong method" from "no such route" for a friendlier message.
        for (const auto & kv : routes_) {
            if (kv.first.size() > req.method.size() &&
                kv.first.compare(req.method.size() + 1, std::string::npos, req.path) == 0) {
                return send_error(w, 405, "method " + req.method + " not allowed on " + req.path);
            }
        }
        return send_error(w, 404, "no route for " + req.path);
    }
    return it->second(req, w);
}

} // namespace http
} // namespace amp
