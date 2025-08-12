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

from pathlib import Path

import torch
from accelerate import Accelerator
from hydra import main
from omegaconf import DictConfig

CONFIGS_PATH: Path = Path(__file__).parent.parent.parent.parent / "configs"


@main(version_base="1.3", config_path=str(CONFIGS_PATH), config_name="hello")
def main(cfg: DictConfig) -> None:  # noqa: D103
    accelerator: Accelerator = Accelerator()
    x: torch.Tensor = torch.ones((5,)) * cfg.constant
    x = x.to(accelerator.device)
    print(x)


if __name__ == "__main__":
    main()
