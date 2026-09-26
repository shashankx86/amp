#include "amp/io/mapped_file.h"

#include "amp/format.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <vector>
#include <cstring>
#include <utility>

namespace amp {

MappedFile::~MappedFile() {
    if (data_) {
        munmap(const_cast<void *>(data_), size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

MappedFile::MappedFile(MappedFile && other) noexcept
    : data_(other.data_), size_(other.size_), fd_(other.fd_), path_(std::move(other.path_)) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.fd_   = -1;
}

MappedFile & MappedFile::operator=(MappedFile && other) noexcept {
    if (this != &other) {
        if (data_) {
            munmap(const_cast<void *>(data_), size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
        data_ = other.data_;
        size_ = other.size_;
        fd_   = other.fd_;
        path_ = std::move(other.path_);
        other.data_ = nullptr;
        other.size_ = 0;
        other.fd_   = -1;
    }
    return *this;
}

Result<MappedFile> MappedFile::open_read(const std::string & path) {
    MappedFile m;
    m.path_ = path;

    m.fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (m.fd_ < 0) {
        return Status::Errorf("open('%s') failed: %s", path.c_str(), strerror(errno));
    }
    struct stat st;
    if (fstat(m.fd_, &st) != 0) {
        const int e = errno;
        close(m.fd_);
        m.fd_ = -1;
        return Status::Errorf("fstat('%s') failed: %s", path.c_str(), strerror(e));
    }
    m.size_ = (size_t) st.st_size;
    if (m.size_ == 0) {
        return Status::Errorf("'%s' is empty", path.c_str());
    }
    // MAP_POPULATE would fault the whole 14.65 GB in and is exactly what we do NOT
    // want at open time; residency is managed deliberately by the warmer instead.
    void * p = mmap(nullptr, m.size_, PROT_READ, MAP_PRIVATE, m.fd_, 0);
    if (p == MAP_FAILED) {
        const int e = errno;
        close(m.fd_);
        m.fd_ = -1;
        return Status::Errorf("mmap('%s', %zu bytes) failed: %s", path.c_str(), m.size_, strerror(e));
    }
    m.data_ = (const uint8_t *) p;
    return m;
}

Status MappedFile::pread_exact(uint64_t offset, void * dst, size_t len) const {
    uint8_t * out = (uint8_t *) dst;
    size_t    done = 0;
    while (done < len) {
        const ssize_t n = pread(fd_, out + done, len - done, (off_t) (offset + done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return Status::Errorf("pread(off=%llu) failed: %s",
                                  (unsigned long long) (offset + done), strerror(errno));
        }
        if (n == 0) {
            return Status::Errorf("pread(off=%llu) hit EOF", (unsigned long long) (offset + done));
        }
        done += (size_t) n;
    }
    return Status::OK();
}

Result<double> MappedFile::resident_fraction(uint64_t offset, uint64_t length) const {
    if (!data_ || length == 0) {
        return 0.0;
    }
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return 0.0;
    }
    const uint64_t start = (offset / (uint64_t) page) * (uint64_t) page;
    const uint64_t end   = std::min<uint64_t>(size_, ((offset + length + page - 1) / page) * page);
    if (end <= start) {
        return 0.0;
    }
    const uint64_t chunk = 4ull * 1024 * 1024;
    const size_t   npages_chunk = (size_t) (chunk / (uint64_t) page);
    std::vector<unsigned char> vec(npages_chunk);
    uint64_t resident = 0, total = 0;
    for (uint64_t off = start; off < end; off += chunk) {
        const uint64_t len = std::min<uint64_t>(chunk, end - off);
        const size_t   n   = (size_t) (len / (uint64_t) page);
        const auto *   addr = (const uint8_t *) data_ + off;
        if (mincore(const_cast<void *>((const void *) addr), len, vec.data()) != 0) {
            return resident / (double) std::max<uint64_t>(1, total);
        }
        for (size_t i = 0; i < n; i++) {
            total++;
            if (vec[i] & 1u) {
                resident++;
            }
        }
    }
    return total ? (double) resident / (double) total : 0.0;
}

Status MappedFile::advise_willneed(uint64_t offset, uint64_t len) const {
    const int rc = posix_fadvise(fd_, (off_t) offset, (off_t) len, POSIX_FADV_WILLNEED);
    return rc == 0 ? Status::OK() : Status::Errorf("posix_fadvise(WILLNEED) rc=%d", rc);
}

Status MappedFile::advise_random(uint64_t offset, uint64_t len) const {
    const int rc = posix_fadvise(fd_, (off_t) offset, (off_t) len, POSIX_FADV_RANDOM);
    return rc == 0 ? Status::OK() : Status::Errorf("posix_fadvise(RANDOM) rc=%d", rc);
}

Status MappedFile::drop_cache(uint64_t offset, uint64_t len) const {
    const int rc = posix_fadvise(fd_, (off_t) offset, (off_t) len, POSIX_FADV_DONTNEED);
    return rc == 0 ? Status::OK() : Status::Errorf("posix_fadvise(DONTNEED) rc=%d", rc);
}

} // namespace amp
