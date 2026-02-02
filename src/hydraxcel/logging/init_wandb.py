# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HydraXcel
# Author: Carel van Niekerk, Benjamin Ruppik
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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Initialise the weights and biases logging."""

import logging
import os
from pathlib import Path

import wandb
import weave  # noqa: F401 # Used by WANDB integration
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from omegaconf import DictConfig, OmegaConf

from hydraxcel.logging.helpers import find_project_root

__all__ = ["initialize_wandb"]


def initialize_wandb(
    *,
    config: DictConfig | None = None,
    project_name: str = "ConfidentLLM",
    accelerator: Accelerator | None = None,
) -> None:
    """Initialize wandb."""
    os.environ["WANDB_SILENT"] = "true"

    wandb_path = find_project_root(Path(__file__)) / "wandb_logs"
    wandb_run: wandb.Run = wandb.init(
        project=project_name,
        dir=wandb_path.as_posix(),
        settings=wandb.Settings(
            start_method="thread",  # Note: https://docs.wandb.ai/guides/integrations/hydra#troubleshooting-multiprocessing
            x_service_wait=300,
            init_timeout=300,
        ),
    )

    if accelerator is not None:
        accelerate_tracker = WandBTracker(project_name)
        accelerate_tracker.run = wandb_run
        accelerator.trackers.append(accelerate_tracker)

    # Clear wandb's internal logging to avoid duplicate messages
    wandb_logger = logging.getLogger("wandb")
    wandb_logger.handlers.clear()
    wandb_logger.propagate = True

    # Get user info and log it
    user_info = wandb.Api().viewer
    team_name = f"({user_info.teams[0]})" if user_info.teams else ""

    msg: str = (
        f"Currently logged in as: {user_info.username}{team_name} to "
        "https://api.wandb.ai. Use `wandb login --relogin` to force relogin"
    )
    wandb_logger.info(msg)
    wandb_logger.info(f"Tracking run with wandb version {wandb.__version__}")  # noqa: G004
    wandb_logger.info(f"Run data is saved locally in {wandb_run.dir}")  # noqa: G004
    wandb_logger.info("Run `wandb offline` to turn off syncing.")
    wandb_logger.info(f"Syncing run {wandb_run.name}")  # noqa: G004
    wandb_logger.info(f"⭐️ View project at {wandb_run.get_project_url()}")  # noqa: G004
    wandb_logger.info(f"🚀 View run at {wandb_run.get_url()}")  # noqa: G004

    if config is None:
        return
    cfg_dict = OmegaConf.to_container(
        config,
        resolve=True,
    )

    # Add the run config to the wandb config
    # (so that they are tracked in the wandb run)
    wandb.config.run = cfg_dict
