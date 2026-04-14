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
    from deepspeed.utils import (  # ty:ignore[unresolved-import]
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
    """Return a standard Python logger, optionally silencing third-party loggers.

    Clears the handlers of the Transformers and (when available) DeepSpeed
    loggers and enables propagation so their messages flow through Hydra's
    configured root handler rather than being emitted twice or swallowed.

    Args:
        name: Logger name; typically ``__name__`` or a script stem.
        setup_transformers_logger: When ``True``, redirect the HuggingFace
            Transformers logger to the root handler.
        setup_deepspeed_logger: When ``True`` and DeepSpeed is installed,
            redirect the DeepSpeed logger to the root handler.

    Returns:
        The ``logging.Logger`` instance for *name*.

    """
    logger: logging.Logger = logging.getLogger(name)

    if setup_transformers_logger:
        # Get the transformer logger and propagate its logs to the Hydra root.
        transformers_logger: logging.Logger = transformers.utils.logging.get_logger()  # ty:ignore[possibly-missing-submodule]
        transformers_logger.handlers.clear()
        transformers_logger.propagate = True

    if setup_deepspeed_logger and SETUP_DEEPSPEED_LOGGER:
        deepspeed_logger.handlers.clear()
        deepspeed_logger.propagate = True

    return logger


def init_logging_platform(  # noqa: PLR0913
    platform: LoggingPlatform,
    config: DictConfig,
    project_name: str,
    task_name: str,
    job_name: str | None = None,
    accelerator: Accelerator | None = None,
) -> None:
    """Initialise the chosen experiment-tracking platform for the current run.

    No-ops when ``platform`` is ``LOCAL``, when running on a non-main process,
    when ``ACCELERATE_DEBUG_MODE`` is set, or when a cluster-submission
    launcher is active (to avoid logging from the sweep coordinator process).

    Args:
        platform: Which tracking backend to initialise.
        config: The resolved Hydra run configuration to log as hyperparameters.
        project_name: Top-level project/experiment name for the tracking
            platform.
        task_name: Sub-task or run-type name (appended to *project_name* for
            W&B; used as the MLflow run name).
        job_name: Optional run/job name derived from config values (passed to
            W&B as the run display name).
        accelerator: The Accelerate ``Accelerator`` instance; used to check
            whether the current process is the main process.

    """
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
            job_name=job_name,
        )
    elif platform == LoggingPlatform.MLFLOW:
        initialize_mlflow(
            config=config,
            experiment_name=project_name,
            run_name=task_name,
            accelerator=accelerator,
        )
