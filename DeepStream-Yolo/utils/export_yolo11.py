import os
import sys
import onnx
import torch
import torch.nn as nn
from copy import deepcopy

import ultralytics.utils
import ultralytics.models.yolo
import ultralytics.utils.tal as _m

sys.modules['ultralytics.yolo'] = ultralytics.models.yolo
sys.modules['ultralytics.yolo.utils'] = ultralytics.utils


def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)


_m.dist2bbox.__code__ = _dist2bbox.__code__


def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


class DeepStreamBoxOutput(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes or 0)

    def forward(self, x):
        tensor = _first_tensor(x)
        if tensor is None:
            raise TypeError("DeepStreamBoxOutput expected tensor-like data")
        tensor = tensor.transpose(1, 2)
        boxes = tensor[:, :, :4]
        tail = tensor[:, :, 4:]
        if tail.shape[-1] == 0:
            scores = torch.ones_like(boxes[:, :, :1])
            labels = torch.zeros_like(scores)
        elif self.num_classes > 0 and tail.shape[-1] >= (1 + self.num_classes):
            obj = tail[:, :, :1]
            cls_logits = tail[:, :, 1 : 1 + self.num_classes]
            cls_scores, labels = torch.max(cls_logits, dim=-1, keepdim=True)
            scores = cls_scores * obj
        else:
            scores, labels = torch.max(tail, dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


class DeepStreamPoseOutput(nn.Module):
    def __init__(self, num_classes: int, kpt_shape):
        super().__init__()
        self.num_classes = int(num_classes or 0)
        nk, dims = kpt_shape if kpt_shape else (0, 0)
        self.kpt_dims = int(nk * dims)

    def forward(self, x):
        tensor = _first_tensor(x)
        if tensor is None:
            raise TypeError("DeepStreamPoseOutput expected tensor-like data")
        tensor = tensor.transpose(1, 2)
        boxes = tensor[:, :, :4]
        obj = tensor[:, :, 4:5]
        offset = 5
        scores = obj
        labels = torch.zeros_like(obj)
        if self.num_classes > 0 and tensor.shape[-1] >= offset + self.num_classes:
            cls_logits = tensor[:, :, offset : offset + self.num_classes]
            offset += self.num_classes
            cls_scores, labels = torch.max(cls_logits, dim=-1, keepdim=True)
            scores = cls_scores * obj
        if self.kpt_dims:
            kpts = tensor[:, :, offset : offset + self.kpt_dims]
        else:
            kpts = tensor[:, :, 0:0]
        return torch.cat([boxes, scores, labels.to(boxes.dtype), kpts], dim=-1)


def yolo11_export(weights, device, inplace=True, fuse=True):
    ckpt = torch.load(weights, map_location='cpu')
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()
    if not hasattr(ckpt, 'stride'):
        ckpt.stride = torch.tensor([32.])
    if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
        ckpt.names = dict(enumerate(ckpt.names))
    model = ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval()
    for m in model.modules():
        t = type(m)
        if hasattr(m, 'inplace'):
            m.inplace = inplace
        elif t.__name__ == 'Upsample' and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    model = deepcopy(model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for k, m in model.named_modules():
        if m.__class__.__name__ in ('Detect', 'RTDETRDecoder'):
            m.dynamic = False
            m.export = True
            m.format = 'onnx'
    return model


def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ResourceWarning)


def main(args):
    suppress_warnings()

    print(f'\nStarting: {args.weights}')

    print('Opening YOLO11 model')

    device = torch.device('cpu')
    model = yolo11_export(args.weights, device)

    if len(model.names.keys()) > 0:
        print('Creating labels.txt file')
        with open('labels.txt', 'w', encoding='utf-8') as f:
            for name in model.names.values():
                f.write(f'{name}\n')

    head = None
    if hasattr(model, 'model'):
        try:
            head = model.model[-1]
        except Exception:
            head = None

    if head is not None and hasattr(head, 'kpt_shape'):
        model = nn.Sequential(model, DeepStreamPoseOutput(getattr(head, 'nc', 0), getattr(head, 'kpt_shape', None)))
    else:
        num_classes = getattr(head, 'nc', 0) if head is not None else len(getattr(model, 'names', {}))
        model = nn.Sequential(model, DeepStreamBoxOutput(num_classes))

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = f'{args.weights}.onnx'

    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'output': {
            0: 'batch'
        }
    }

    print('Exporting the model to ONNX')
    torch.onnx.export(
        model, onnx_input_im, onnx_output_file, verbose=False, opset_version=args.opset, do_constant_folding=True,
        input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        print('Simplifying the ONNX model')
        import onnxslim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f'Done: {onnx_output_file}\n')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DeepStream YOLO11 conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
