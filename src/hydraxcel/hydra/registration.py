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
"""Plugin for Hydra registration."""

from dataclasses import field
from typing import Protocol

from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugin, Plugins

__all__ = ["register_plugin"]


class Configuration(Protocol):
    """Protocol for hydra configuration."""

    _target_: str = field(
        default="hydra_plugins.hydra.launcher.basic_launcher.BasicLauncher",
        metadata={"help": "Target class to instantiate"},
    )


def register_plugin(
    plugin_name: str,
    config: Configuration,
    plugin_class: type[Plugin],
) -> None:
    """Register the plugin."""
    if not config._target_.startswith("hydra_plugins."):
        config._target_ = f"hydra_plugins.{config._target_}"

    ConfigStore.instance().store(
        group="hydra/launcher",
        name=plugin_name,
        node=config,
    )
    plugins_manager: Plugins = Plugins.instance()
    plugins_manager.class_name_to_class[config._target_] = plugin_class
