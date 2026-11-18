#pragma once

// One replaceable latest frame, shared only with optional live processing.
// Neither participant waits for the other's lock. A dead reader releases its
// flock automatically; raw acquisition never depends on reader progress.
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace squeakview {
constexpr uint64_t LIVE_MAGIC = 0x53514c4956450001ULL;
constexpr size_t LIVE_METADATA_CAPACITY = 65536;
constexpr size_t LIVE_HEADER_CAPACITY = 4096;
constexpr size_t LIVE_MAX_PIXELS = 64 * 1024 * 1024;
struct LiveHeader {
  uint64_t magic, sequence, closed, width, height, pts, duration, pixels, metadata;
};
struct LiveFrame {
  LiveHeader header{};
  std::vector<uint8_t> pixels;
  std::string metadata;
};
class LiveMailbox {
  int fd_ = -1;
  uint8_t* data_ = nullptr;
  size_t size_ = 0;
public:
  explicit LiveMailbox(const char* path) {
    fd_ = ::open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
    struct stat info{};
    if (fd_ < 0 || fstat(fd_, &info) || !S_ISREG(info.st_mode) ||
        info.st_size < static_cast<off_t>(LIVE_HEADER_CAPACITY + LIVE_METADATA_CAPACITY + 1) ||
        info.st_size > static_cast<off_t>(LIVE_HEADER_CAPACITY + LIVE_METADATA_CAPACITY + LIVE_MAX_PIXELS)) {
      if (fd_ >= 0) ::close(fd_);
      throw std::runtime_error("invalid live frame mailbox");
    }
    size_ = static_cast<size_t>(info.st_size);
    void* mapped = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapped == MAP_FAILED) {
      ::close(fd_);
      throw std::runtime_error("cannot map live frame mailbox");
    }
    data_ = static_cast<uint8_t*>(mapped);
  }
  ~LiveMailbox() { munmap(data_, size_); ::close(fd_); }
  LiveMailbox(const LiveMailbox&) = delete;
  LiveMailbox& operator=(const LiveMailbox&) = delete;
  bool publish(const uint8_t* pixels, size_t bytes, const std::string& metadata,
               uint64_t width, uint64_t height, uint64_t pts, uint64_t duration) {
    if (bytes > size_ - LIVE_HEADER_CAPACITY - LIVE_METADATA_CAPACITY ||
        metadata.size() > LIVE_METADATA_CAPACITY || flock(fd_, LOCK_EX | LOCK_NB)) return false;
    LiveHeader old{};
    std::memcpy(&old, data_, sizeof(old));
    LiveHeader header{LIVE_MAGIC, old.sequence + 1, 0, width, height, pts, duration,
                      static_cast<uint64_t>(bytes), static_cast<uint64_t>(metadata.size())};
    std::memcpy(data_ + LIVE_HEADER_CAPACITY, metadata.data(), metadata.size());
    std::memcpy(data_ + LIVE_HEADER_CAPACITY + LIVE_METADATA_CAPACITY, pixels, bytes);
    std::memcpy(data_, &header, sizeof(header));
    flock(fd_, LOCK_UN);
    return true;
  }
  bool read(uint64_t previous, LiveFrame& frame, bool& closed) {
    if (flock(fd_, LOCK_SH | LOCK_NB)) return false;
    LiveHeader header{};
    std::memcpy(&header, data_, sizeof(header));
    closed = header.magic == LIVE_MAGIC && header.closed;
    const bool valid = header.magic == LIVE_MAGIC && header.sequence > previous &&
        header.width > 0 && header.height > 0 && header.width <= 8192 && header.height <= 8192 &&
        header.pixels == header.width * header.height && header.metadata <= LIVE_METADATA_CAPACITY &&
        header.pixels <= size_ - LIVE_HEADER_CAPACITY - LIVE_METADATA_CAPACITY;
    try {
      if (valid) {
        frame.header = header;
        frame.metadata.assign(reinterpret_cast<char*>(data_ + LIVE_HEADER_CAPACITY), header.metadata);
        const auto* begin = data_ + LIVE_HEADER_CAPACITY + LIVE_METADATA_CAPACITY;
        frame.pixels.assign(begin, begin + header.pixels);
      }
    } catch (...) { flock(fd_, LOCK_UN); throw; }
    flock(fd_, LOCK_UN);
    return valid;
  }
  void finish() {
    if (flock(fd_, LOCK_EX | LOCK_NB)) return;
    LiveHeader header{};
    std::memcpy(&header, data_, sizeof(header));
    header.magic = LIVE_MAGIC;
    header.closed = 1;
    std::memcpy(data_, &header, sizeof(header));
    flock(fd_, LOCK_UN);
  }
};
}  // namespace squeakview
