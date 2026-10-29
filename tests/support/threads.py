"""Worker failures must reach the owning unittest, with bounded joins."""
import threading


class CheckedThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._error = None

    def run(self):
        try:
            super().run()
        except BaseException as error:
            self._error = error

    def join(self, timeout=5):
        super().join(timeout)
        if self.is_alive():
            raise TimeoutError(f"test worker {self.name} did not finish within {timeout}s")
        if self._error is not None:
            raise self._error
