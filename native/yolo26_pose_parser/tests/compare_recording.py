"""Bounded diagnostic: compare both parser callbacks on four real engine tensors.

This is not the offline re-inference pipeline or acquisition validation. It reads
four sampled video frames, executes the existing engine, and compares callbacks
on identical tensors. It writes temporary tensors only; no model/run mutation.
Run with the acquisition app idle.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--compare', type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import tensorrt as trt
    import torch

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError('engine deserialization failed')
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('expected one input/output')
    input_name, output_name = inputs[0], outputs[0]
    input_shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))
    if input_shape != (1, 3, 640, 640) or output_shape != (1, 300, 63):
        raise ValueError(f'diagnostic expects baseline model shapes, got {input_shape}, {output_shape}')
    if any(engine.get_tensor_dtype(n) != trt.DataType.FLOAT for n in names):
        raise ValueError('diagnostic expects FLOAT I/O')
    stream = torch.cuda.Stream()
    video = cv2.VideoCapture(str(args.video))
    try:
        with tempfile.TemporaryDirectory(prefix='pose-parser-tensors-') as temp:
            for ordinal in (0, 300, 600, 900):
                if not video.set(cv2.CAP_PROP_POS_FRAMES, ordinal):
                    raise RuntimeError(f'cannot seek to frame {ordinal}')
                ok, image = video.read()
                if not ok:
                    raise RuntimeError(f'cannot decode sampled frame {ordinal}')
                height, width = image.shape[:2]
                gain = min(640 / width, 640 / height)
                resized = cv2.resize(image, (round(width * gain), round(height * gain)))
                canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
                y, x = (640 - resized.shape[0]) // 2, (640 - resized.shape[1]) // 2
                canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                array = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
                with torch.cuda.stream(stream):
                    tensor = torch.from_numpy(array).to(device='cuda')
                    result = torch.empty(output_shape, dtype=torch.float32, device='cuda')
                    context.set_tensor_address(input_name, tensor.data_ptr())
                    context.set_tensor_address(output_name, result.data_ptr())
                    if not context.execute_async_v3(stream.cuda_stream):
                        raise RuntimeError('TensorRT execution failed')
                stream.synchronize()
                output = Path(temp) / 'output.f32'
                result.cpu().numpy().tofile(output)
                env = dict(os.environ, SQUEAKVIEW_SRC_W=str(width), SQUEAKVIEW_SRC_H=str(height))
                process = subprocess.run(
                    [str(args.compare.resolve()), str(args.baseline.resolve()), str(args.candidate.resolve()),
                     str(output), '300', '63'], env=env, capture_output=True, text=True, timeout=20,
                )
                if process.returncode:
                    raise RuntimeError(process.stdout + process.stderr)
                print(f'Frame {ordinal}: {process.stdout.strip()}', flush=True)
    finally:
        video.release()


if __name__ == '__main__':
    main()
