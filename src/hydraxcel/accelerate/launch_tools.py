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

import subprocess
import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from accelerate.commands.launch import launch_command
from hydra import main
from hydra.core.config_store import ConfigStore

from hydraxcel.accelerate.config import LaunchConfig
from hydraxcel.hydra import flatten_config
from hydraxcel.run.setup import _setup_hydra_config_and_logging

__all__ = ["launch"]
_config_store = ConfigStore.instance()
_config_store.store(name="launch_config", node=LaunchConfig)


def _format_multirun_launch_args(
    script: str,
    launch_args: list[str] | None = None,
) -> list[str]:
    formatted_launch_args: list[str] = [f"+launch.script={script}"]
    if launch_args is None:
        return formatted_launch_args
    for arg in launch_args:
        arg_name, arg_value = arg.split("=", 1)
        delimiter = "." if "/" not in arg_name else "/"
        arg_name = delimiter.join(["+launch", arg_name])
        formatted_launch_args.append(f"{arg_name}={arg_value}")
    return formatted_launch_args


def _extract_pass_through_args() -> list[str]:
    """Extract passthrough arguments from sys.argv."""
    is_multirun: bool = False
    if "-m" in sys.argv:
        sys.argv.pop(sys.argv.index("-m"))
        is_multirun = True
    if "--multirun" in sys.argv:
        sys.argv.pop(sys.argv.index("--multirun"))
        is_multirun = True
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        passthrough: list[str] = sys.argv[1:idx]
        if is_multirun:
            passthrough = ["-m", *passthrough]
            launch_args = _format_multirun_launch_args(sys.argv[0], sys.argv[idx + 1 :])
            passthrough.extend(launch_args)
        sys.argv = (
            [sys.argv[0], *sys.argv[idx + 1 :]] if not is_multirun else [sys.argv[0]]
        )
        return passthrough
    if len(sys.argv) > 1:
        passthrough = sys.argv[1:]
        if is_multirun:
            passthrough = ["-m", *passthrough]
            launch_args = _format_multirun_launch_args(sys.argv[0])
            passthrough.extend(launch_args)
        sys.argv = [sys.argv[0]]
        return passthrough
    if is_multirun:
        return ["-m", *_format_multirun_launch_args(sys.argv[0])]
    return []


def launch(
    script_path: Path,
    *,
    hydra_configs_dir: str | None = None,
    config_name: str = "accelerate",
    hydra_base_version: str = "1.4",
) -> Callable[[LaunchConfig], None]:
    """Launch a script at a given path."""
    passthrough_args = _extract_pass_through_args()

    _setup_hydra_config_and_logging(
        file_path=Path("accelerate"),
        config_keys=[],
        change_to_output_dir=False,
    )

    @main(
        config_path=hydra_configs_dir,
        config_name=config_name,
        version_base=hydra_base_version,
    )
    def launch_fn(cfg: LaunchConfig) -> None:
        """Run the main entry point for the script."""
        if hasattr(cfg, "training_script") and cfg.training_script:
            raise ValueError(  # noqa: TRY003
                "Training script is already set, will be overwritten.",  # noqa: EM101
            )
        if not hasattr(cfg, "training_script"):
            msg: str = (
                "When using dataclass config for launching, the training_script"
                "and training_script_args must be set to '' and [] respectively."
            )
            raise ValueError(msg)

        cfg.training_script = script_path.as_posix()
        cfg.training_script_args = passthrough_args

        # Flatten and validate the configuration
        cfg: dict = flatten_config(cfg)  # type: ignore[invalid-argument-type]
        for key in [
            "deepspeed_fields_from_accelerate_config",
            "use_cpu",
            "use_xpu",
            "nproc_per_node",
            "master_port",
        ]:
            cfg.pop(key, None)

        cfg = asdict(LaunchConfig(**cfg))
        cfg: Namespace = Namespace(**cfg)

        # If -m is in the passthrough args, run the script directly
        if "-m" in passthrough_args or "--help" in passthrough_args:
            cmd = ["uv", "run", script_path.as_posix(), *passthrough_args]
            subprocess.run(cmd, check=True)  # noqa: S603
            sys.exit(0)
        launch_command(cfg)

    return launch_fn
