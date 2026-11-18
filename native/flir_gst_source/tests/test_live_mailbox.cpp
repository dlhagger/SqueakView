// Hardware-independent fault tests for the production latest-frame transport.
#include "../src/live_mailbox.hpp"
#include <cassert>
#include <chrono>
#include <csignal>
#include <sys/wait.h>

int main() {
  char path[] = "/tmp/squeakview-mailbox-test-XXXXXX";
  int file = mkstemp(path);
  assert(file >= 0);
  assert(posix_fallocate(file, 0, squeakview::LIVE_HEADER_CAPACITY +
                         squeakview::LIVE_METADATA_CAPACITY + 4) == 0);
  close(file);
  {
    squeakview::LiveMailbox writer(path), reader(path);
    uint8_t pixels[] = {1, 2, 3, 4};
    assert(writer.publish(pixels, 4, "{\"source_sequence_index\":0}", 2, 2, 123, 33));
    squeakview::LiveFrame frame;
    bool closed = false;
    assert(reader.read(0, frame, closed));
    assert(!closed && frame.header.pts == 123 && frame.header.duration == 33);
    assert(frame.pixels == std::vector<uint8_t>({1, 2, 3, 4}));
    assert(frame.metadata == "{\"source_sequence_index\":0}");
    assert(!reader.read(frame.header.sequence, frame, closed));

    // A consumer can keep its read lock forever (SIGSTOP, deadlock, debugger).
    // The recording producer still returns immediately and drops only export.
    int ready[2];
    assert(pipe(ready) == 0);
    pid_t child = fork();
    assert(child >= 0);
    if (!child) {
      close(ready[0]);
      int fd = open(path, O_RDWR);
      if (fd < 0 || flock(fd, LOCK_SH)) _exit(2);
      if (write(ready[1], "1", 1) != 1) _exit(3);
      while (true) pause();
    }
    close(ready[1]);
    char ack = 0;
    assert(read(ready[0], &ack, 1) == 1 && ack == '1');
    close(ready[0]);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i)
      assert(!writer.publish(pixels, 4, "skipped", 2, 2, 456, 33));
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
    assert(kill(child, SIGKILL) == 0);
    int status;
    assert(waitpid(child, &status, 0) == child && WIFSIGNALED(status));
    assert(writer.publish(pixels, 4, "after crash", 2, 2, 789, 33));
    assert(reader.read(frame.header.sequence, frame, closed));
    assert(frame.header.pts == 789 && frame.metadata == "after crash");

    // Intermediate exports are replaceable; identity/PTS are never renumbered.
    assert(writer.publish(pixels, 4, "old", 2, 2, 1000, 33));
    assert(writer.publish(pixels, 4, "latest", 2, 2, 9000, 33));
    assert(reader.read(frame.header.sequence, frame, closed));
    assert(frame.header.pts == 9000 && frame.metadata == "latest");
    assert(!writer.publish(pixels, 5, "oversized", 2, 2, 10000, 33));
    assert(!writer.publish(pixels, 4, std::string(65537, 'x'), 2, 2, 10000, 33));
    writer.finish();
    assert(!reader.read(frame.header.sequence, frame, closed) && closed);
  }
  unlink(path);
  return 0;
}
