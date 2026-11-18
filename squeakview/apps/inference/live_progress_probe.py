"""Count completed optional output without retaining SDK buffers or metadata."""
from pyservicemaker import BatchMetadataOperator, BufferOperator, Probe

from .callback_guard import NativeCallbackGuard


def attach_live_progress(pipeline, config, counts, on_fatal):
    if config.preview_sockets:
        for stream in range(config.num_cameras):
            pipeline.attach(f'preview_queue{stream}', Probe(f'live_progress{stream}',
                LiveBufferProgress(counts, stream, on_fatal)))
    else:
        pipeline.attach('tracker' if config.enable_infer else 'mux',
                        Probe('live_progress', LiveBatchProgress(counts, on_fatal)))


class LiveBufferProgress(BufferOperator):
    def __init__(self, counts, stream, on_fatal):
        super().__init__()
        self.counts, self.stream = counts, stream
        self.guard = NativeCallbackGuard('live output progress', on_fatal)

    def handle_buffer(self, _buffer):
        return self.guard.call(self._advance)

    def _advance(self):
        self.counts[self.stream] += 1


class LiveBatchProgress(BatchMetadataOperator):
    def __init__(self, counts, on_fatal):
        super().__init__()
        self.counts = counts
        self.guard = NativeCallbackGuard('live output progress', on_fatal)

    def handle_metadata(self, batch):
        self.guard.call(lambda: self._advance(batch))

    def _advance(self, batch):
        for frame in batch.frame_items:
            source = getattr(frame, 'source_id', None)
            stream = int(frame.pad_index if source is None else source)
            if not 0 <= stream < len(self.counts):
                raise ValueError(f'unexpected live source {stream}')
            self.counts[stream] += 1
