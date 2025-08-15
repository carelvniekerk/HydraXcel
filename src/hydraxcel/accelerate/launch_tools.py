# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HydraXcel
# Author: Carel van Niekerk
# Year: 2025
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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
"""Custom launcher for using accelerate in uv script commands."""

import sys
from argparse import Namespace
from pathlib import Path
from typing import Callable

from accelerate.commands.launch import launch_command
from hydra import main
from hydra.core.config_store import ConfigStore

from hydraxcel.accelerate.config import LaunchConfig

__all__ = ["launch"]
_config_store = ConfigStore.instance()
_config_store.store(name="launch_config", node=LaunchConfig)


def _extract_pass_through_args() -> list[str]:
    """Extract passthrough arguments from sys.argv."""
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        passthrough: list[str] = sys.argv[1:idx]
        sys.argv = [sys.argv[0], *sys.argv[idx + 1 :]]
        return passthrough
    if len(sys.argv) > 1:
        passthrough = sys.argv[1:]
        sys.argv = [sys.argv[0]]
        return passthrough
    return []


def launch(
    script_path: Path,
    *,
    hydra_configs_dir: str,
    config_name: str = "accelerate",
    hydra_base_version: str = "1.3",
) -> Callable[[LaunchConfig], None]:
    """Launch a script at a given path."""
    passthrough_args = _extract_pass_through_args()

    @main(
        config_path=hydra_configs_dir,
        config_name=config_name,
        version_base=hydra_base_version,
    )
    def launch_fn(cfg: LaunchConfig) -> None:
        """Run the main entry point for the script."""
        cfg.training_script = str(script_path)
        cfg.training_script_args = passthrough_args

        launch_command(Namespace(**cfg))

    return launch_fn
