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
"""Logging initialization for HydraXcel."""

import logging
import os
from enum import StrEnum, auto
from pathlib import Path

import transformers
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf

from hydraxcel.logging.init_mlflow import initialize_mlflow
from hydraxcel.logging.init_wandb import initialize_wandb

try:
    from deepspeed.utils import (  # type: ignore[unresolved-import]
        logger as deepspeed_logger,
    )

    SETUP_DEEPSPEED_LOGGER = True
except ImportError:
    SETUP_DEEPSPEED_LOGGER = False

__all__ = [
    "LoggingPlatform",
    "get_logger",
    "init_logging_platform",
]


class LoggingPlatform(StrEnum):
    """Logging platforms for HydraXcel."""

    LOCAL = auto()
    WANDB = auto()
    MLFLOW = auto()


def get_logger(
    name: str = "__main__",
    *,
    setup_transformers_logger: bool = True,
    setup_deepspeed_logger: bool = SETUP_DEEPSPEED_LOGGER,
) -> logging.Logger:
    """Get the logger."""
    logger: logging.Logger = logging.getLogger(name)

    if setup_transformers_logger:
        # Get the transformer logger and propagate its logs to the Hydra root.
        transformers_logger: logging.Logger = transformers.utils.logging.get_logger()  # type: ignore[unresolved-attribute] # TODO: Remove when ty bug is fixed
        transformers_logger.handlers.clear()
        transformers_logger.propagate = True

    if setup_deepspeed_logger and SETUP_DEEPSPEED_LOGGER:
        deepspeed_logger.handlers.clear()
        deepspeed_logger.propagate = True

    return logger


def init_logging_platform(
    platform: LoggingPlatform,
    config: DictConfig,
    project_name: str,
    task_name: str,
    accelerator: Accelerator | None = None,
) -> None:
    """Initialize the logging platform."""
    if platform == LoggingPlatform.LOCAL:
        return

    if accelerator is not None and not accelerator.is_main_process:
        return
    if os.getenv("ACCELERATE_DEBUG_MODE", default=False):  # noqa: PLW1508
        return

    hydra_config_path: Path = Path(".hydra") / "hydra.yaml"
    hydra_config: DictConfig = OmegaConf.load(hydra_config_path)  # type: ignore  # noqa: PGH003
    if (
        hydra_config.hydra.mode.lower() == "multirun"
        and "submission" in config.hydra.launcher.__target__
    ):
        return

    if platform == LoggingPlatform.WANDB:
        project_name: str = f"{project_name}-{task_name}"
        initialize_wandb(
            config=config,
            project_name=project_name,
            accelerator=accelerator,
        )
    elif platform == LoggingPlatform.MLFLOW:
        initialize_mlflow(
            config=config,
            experiment_name=project_name,
            run_name=task_name,
            accelerator=accelerator,
        )
