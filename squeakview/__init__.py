"""Shim package to expose the existing modules under the `squeakview` namespace."""

from importlib import import_module
import sys as _sys

_ALIASES = ("config", "apps", "common", "services", "tests")

for _name in _ALIASES:
    try:
        _mod = import_module(_name)
    except Exception:
        continue
    _sys.modules[f"{__name__}.{_name}"] = _mod
