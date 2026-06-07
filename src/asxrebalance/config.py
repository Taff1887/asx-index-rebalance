"""YAML config loader and environment-variable handling."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR


def _load_yaml(name: str) -> dict[str, Any]:
    path: Path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=None)
def load_indices_config() -> dict[str, Any]:
    return _load_yaml("indices.yaml")


@lru_cache(maxsize=None)
def load_methodology_config() -> dict[str, Any]:
    return _load_yaml("methodology.yaml")


@lru_cache(maxsize=None)
def load_data_sources_config() -> dict[str, Any]:
    return _load_yaml("data_sources.yaml")


@lru_cache(maxsize=None)
def load_validation_config() -> dict[str, Any]:
    return _load_yaml("validation.yaml")


@lru_cache(maxsize=None)
def load_strategy_config() -> dict[str, Any]:
    return _load_yaml("strategy.yaml")


@lru_cache(maxsize=None)
def load_costs_config() -> dict[str, Any]:
    return _load_yaml("costs.yaml")


def get_env(name: str, default: str | None = None) -> str | None:
    """Return an env var, with optional default. Does not raise on missing."""
    return os.environ.get(name, default)


def reset_config_cache() -> None:
    """Clear cached configs. Used in tests after editing YAML."""
    load_indices_config.cache_clear()
    load_methodology_config.cache_clear()
    load_data_sources_config.cache_clear()
    load_validation_config.cache_clear()
    load_strategy_config.cache_clear()
    load_costs_config.cache_clear()


__all__ = [
    "load_indices_config",
    "load_methodology_config",
    "load_data_sources_config",
    "load_validation_config",
    "load_strategy_config",
    "load_costs_config",
    "get_env",
    "reset_config_cache",
]
