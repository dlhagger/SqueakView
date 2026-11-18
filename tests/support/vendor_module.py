"""Load Python boundary logic with inert SDK bases when the SDK is unavailable.

The isolated alias avoids replacing application modules or leaking fake vendor
modules into other tests. These are Python policy tests, never SDK qualification.
"""
import importlib
import importlib.util
from pathlib import Path
import sys
import types
from unittest import mock


def load_inference_module(name):
    try:
        import pyservicemaker  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != 'pyservicemaker':
            raise
    else:
        return importlib.import_module('squeakview.apps.inference.' + name)

    class Pipeline:
        def __init__(self, *_args): pass
        def __getitem__(self, _name): raise KeyError(_name)
        def add(self, *_args, **_kwargs): return self
        def start(self, *_args): return self
        def stop(self): return self
        def wait(self): return self

    fake = types.ModuleType('pyservicemaker')
    fake.Pipeline = Pipeline
    fake.EOSMessage = type('EOSMessage', (), {})
    fake.BufferOperator = type('BufferOperator', (), {})
    fake.BatchMetadataOperator = type('BatchMetadataOperator', (), {})
    fake.Probe = lambda name, operator: (name, operator)
    path = Path(__file__).resolve().parents[2] / 'squeakview/apps/inference' / f'{name}.py'
    spec = importlib.util.spec_from_file_location('squeakview.apps.inference._test_' + name, path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {'pyservicemaker': fake}):
        spec.loader.exec_module(module)
    return module
