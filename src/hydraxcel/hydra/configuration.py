# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HydraXcel
# Author: Carel van Niekerk
# Year: 2026
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT Codex, Claude Code, Gemini.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tools for hydra and configuration management."""

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
    """Flatten nested configurations for hydra."""
    if isinstance(configuration, DictConfig):
        configuration: dict[str, Any] = OmegaConf.to_container(  # type: ignore[assignment]
            configuration,
            resolve=True,
        )
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
    """Create a Hydra configuration dataclass."""
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
