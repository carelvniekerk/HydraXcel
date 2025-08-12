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
"""Test run script for developing the Accelerate Launcher."""

import os
from pathlib import Path

import torch
from accelerate import Accelerator
from hydra import main
from omegaconf import DictConfig

from agenticrl.scripts.setup_tools import (
    get_logger,
    init_wandb,
    log_system_info,
    setup_hydra_config_and_logging,
)

JOB_NAME = Path(__file__).stem
logger = get_logger(JOB_NAME)
CONFIGS_PATH: Path = Path(__file__).parent.parent.parent.parent / "configs"
setup_hydra_config_and_logging(
    job_name=JOB_NAME,
    config_keys=["constant"],
)


@main(version_base="1.3", config_path=str(CONFIGS_PATH), config_name=JOB_NAME)
def main(cfg: DictConfig) -> None:  # noqa: D103
    log_system_info()
    accelerator: Accelerator = Accelerator()
    if os.environ.get("ACCELERATE_DEBUG_MODE", "0") == "1":
        init_wandb(JOB_NAME, "ConfidentLLM", cfg)
    x: torch.Tensor = torch.ones((5,)) * cfg.constant
    x = x.to(accelerator.device)
    logger.info(x)


if __name__ == "__main__":
    main()
