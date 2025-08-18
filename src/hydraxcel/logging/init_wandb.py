# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HydraXcel
# Author: Carel van Niekerk, Benjamin Ruppik
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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Initialise the weights and biases logging."""

from pathlib import Path

import wandb
from omegaconf import DictConfig, OmegaConf

from hydraxcel.logging.helpers import find_project_root

__all__ = ["initialize_wandb"]


def initialize_wandb(
    *,
    config: DictConfig | None = None,
    project_name: str = "ConfidentLLM",
) -> None:
    """Initialize wandb."""
    wandb_path = find_project_root(Path(__file__))
    wandb.init(
        project=project_name,
        dir=wandb_path.as_posix(),
        settings=wandb.Settings(
            start_method="thread",  # Note: https://docs.wandb.ai/guides/integrations/hydra#troubleshooting-multiprocessing
            x_service_wait=300,
            init_timeout=300,
        ),
    )

    if config is None:
        return
    cfg_dict = OmegaConf.to_container(
        config,
        resolve=True,
    )

    # Add the run config to the wandb config
    # (so that they are tracked in the wandb run)
    wandb.config.run = cfg_dict  # type: ignore[attr-defined]
