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
"""Example script showcasing the use of HydraXcel."""

from pathlib import Path

import torch
from accelerate import Accelerator
from omegaconf import DictConfig

from hydraxcel import get_logger, hydraxcel_main

__all__ = ["main"]
logger = get_logger(Path(__file__).stem)
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@hydraxcel_main(
    "ConfidentLLM",
    hydra_configs_dir=str(CONFIGS_DIR),
    logging_platform="mlflow",
)
def main(cfg: DictConfig, accelerator: Accelerator) -> None:  # noqa: D103
    x: torch.Tensor = torch.ones((5,)) * cfg.constant
    x = x.to(accelerator.device)
    logger.info(x)


if __name__ == "__main__":
    main()
