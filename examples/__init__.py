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
"""Example scripts init for using the accelerate launch wrapper."""

from pathlib import Path

from hydraxcel.accelerate import launch

__all__ = ["hello"]
SCRIPTS_PATH: Path = Path(__file__).parent
CONFIGS_PATH: Path = SCRIPTS_PATH / "configs"

launch_hello = launch(
    SCRIPTS_PATH / "hello.py",
    hydra_configs_dir=str(CONFIGS_PATH),
)

# Only for demonstration purposes, should be run as a module using uv as documented.
if __name__ == "__main__":
    launch_hello()  # type: ignore[missing-argument]
