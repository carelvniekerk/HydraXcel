# coding=utf-8
# --------------------------------------------------------------------------------
# Project: AgenticRL
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
"""Custom launcher for using accelerate."""

import sys
from pathlib import Path
from typing import Callable

from accelerate.commands.launch import launch_command
from hydra import main
from omegaconf import DictConfig

from agenticrl.accelerate.config import CONFIGS_PATH, LaunchConfig

__all__ = ["launch"]


def extract_pass_through_args() -> list[str]:
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


def _launch(
    script_path: Path,
    passthrough_args: list[str],
    hydra_config: DictConfig,
) -> None:
    """Run a script with passthrough arguments."""
    config: LaunchConfig = LaunchConfig.model_validate(hydra_config)
    config.training_script = str(script_path)
    config.training_script_args = passthrough_args

    launch_command(config)


def launch(script_name: str) -> Callable[[DictConfig], None]:
    """Launch a script with the given name."""
    script_path: Path = Path(__file__).parent.parent / "scripts" / f"{script_name}.py"
    passthrough_args = extract_pass_through_args()

    @main(config_path=str(CONFIGS_PATH), config_name="accelerate", version_base="1.3")
    def launch_fn(cfg: DictConfig) -> None:
        """Run the main entry point for the script."""
        _launch(
            script_path=script_path,
            passthrough_args=passthrough_args,
            hydra_config=cfg,
        )

    return launch_fn
