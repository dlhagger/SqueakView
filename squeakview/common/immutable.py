"""Small recursively immutable containers for in-process contracts."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


class FrozenDict(dict):
    """A JSON-serializable dictionary that rejects mutation."""

    @staticmethod
    def _immutable(*_args, **_kwargs):
        raise TypeError("frozen mapping cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, _memo):
        return self


def deep_freeze(value: Any) -> Any:
    """Snapshot and recursively freeze common JSON-like containers."""

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return FrozenDict({key: freeze(child) for key, child in item.items()})
        if isinstance(item, (list, tuple)):
            return tuple(freeze(child) for child in item)
        if isinstance(item, (set, frozenset)):
            return frozenset(freeze(child) for child in item)
        return copy.deepcopy(item)

    # Rebuild containers recursively instead of deepcopying the container
    # first. ``MappingProxyType`` is deliberately not pickleable, but it is a
    # normal immutable mapping at our validated IPC boundary.
    return freeze(value)


__all__ = ["FrozenDict", "deep_freeze"]
