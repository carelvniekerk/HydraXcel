# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HydraXcel
# Author: Carel van Niekerk
# Year: 2026
# --------------------------------------------------------------------------------
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT Codex, Claude Code, Gemini.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for building and flattening Hydra configuration dataclasses."""

from dataclasses import field, make_dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

__all__ = [
    "flatten_config",
    "hydra_config",
]


def flatten_config(
    configuration: DictConfig | dict[str, Any],
    *,
    max_depth: int | None = 5,
) -> dict[str, Any]:
    """Recursively flatten a nested Hydra/OmegaConf configuration into a flat dict.

    Nested sub-configs are merged into the top-level dictionary; leaf values
    are kept as-is.  Used to convert composite Hydra configs (e.g. the
    structured ``AccelerateConfig``) into a flat ``{key: value}`` mapping
    suitable for passing to ``accelerate launch``.

    Args:
        configuration: A Hydra ``DictConfig`` or plain nested dict to flatten.
        max_depth: Maximum recursion depth.  ``None`` means unlimited.
            Defaults to ``5`` to guard against pathological nesting.

    Returns:
        A single-level ``dict`` containing all leaf key-value pairs.

    """
    if isinstance(configuration, DictConfig):
        configuration: dict[str, Any] = OmegaConf.to_container(
            configuration,
            resolve=True,
        )  # ty:ignore[invalid-assignment]
    items: dict[str, Any] = {}
    if max_depth is not None and max_depth < 0:
        return items
    for key, value in configuration.items():
        if isinstance(value, dict):
            items.update(
                flatten_config(
                    configuration=value,
                    max_depth=None if max_depth is None else max_depth - 1,
                ),
            )
        else:
            items[key] = value
    return items


def hydra_config(
    name: str,
    *,
    defaults: list[str | dict[str, str]] | None = None,
    properties: list[tuple[str, Any, Any]] | None = None,
    add_training_script_placeholder: bool = False,
) -> type:
    """Dynamically create a structured Hydra configuration dataclass.

    Generates a ``dataclass`` containing a ``defaults`` list field (for Hydra
    config-group composition) and any additional typed properties.  Optionally
    appends ``training_script`` / ``training_script_args`` placeholders needed
    by the Accelerate launcher.

    Args:
        name: Class name of the generated dataclass.
        defaults: Hydra ``defaults`` list entries (strings or
            ``{group: choice}`` dicts) to embed as the field default factory.
        properties: Extra fields to add as ``(field_name, type, default)``
            triples, forwarded directly to ``make_dataclass``.
        add_training_script_placeholder: When ``True``, appends
            ``training_script: str = ""`` and
            ``training_script_args: list[str] = []`` fields required by the
            Accelerate launch config.

    Returns:
        A freshly created dataclass type registered with Hydra.

    """
    defaults = [] if defaults is None else defaults
    values: list[tuple[str, Any, Any]] = [
        (
            "defaults",
            list[Any],
            field(default_factory=lambda: defaults),
        ),
    ]

    if properties is not None:
        values.extend(properties)

    if add_training_script_placeholder:
        values.append(("training_script", str, field(default="")))
        values.append(("training_script_args", list[str], field(default_factory=list)))

    return make_dataclass(name, values)
