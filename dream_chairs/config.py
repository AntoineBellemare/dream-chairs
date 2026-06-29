"""Tiny YAML-config loader with attribute/dict access."""
from __future__ import annotations

from typing import Any

import yaml


class DotDict(dict):
    """dict with attribute access. Nested dicts are converted to DotDict *in place* at
    construction, so attribute access returns the live object (mutations persist) —
    e.g. `cfg.embedding["backend"] = "openshape"` actually updates the config."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, val in list(self.items()):
            if isinstance(val, dict) and not isinstance(val, DotDict):
                self[key] = DotDict(val)
            elif isinstance(val, list):
                self[key] = [DotDict(v) if isinstance(v, dict) and not isinstance(v, DotDict) else v
                             for v in val]

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str) -> DotDict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return DotDict(data)
