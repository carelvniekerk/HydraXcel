# coding=utf-8
# --------------------------------------------------------------------------------
# Project: Calibrated LLM
# Author: HydraXcel
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
"""Class name resolver for HydraXcel."""

from omegaconf import DictConfig, OmegaConf

__all__ = [
    "class_name_resolver",
]


def class_name_resolver(config_obj: DictConfig) -> str:
    """OmegaConf resolver to get the class name from a config object.

    This resolver takes a config object and returns the class name of the object
    that would be returned by OmegaConf.to_object().

    Args:
        config_obj: The config object (e.g., cfg.trainer, cfg.model)

    Returns:
        The class name of the object that would be returned by OmegaConf.to_object()

    Example:
        # In your config YAML or when using the resolver:
        # trainer_name: ${get_class_name:${trainer}}
        # This would return something like "DPOTrainer"

    """
    # Convert to object to get the class
    obj = OmegaConf.to_object(config_obj)

    # Return the class name
    return obj.__class__.__name__
