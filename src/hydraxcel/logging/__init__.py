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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Set up logging configuration for project."""

from hydraxcel.logging.environment_logging import log_accelerator_info, log_system_info
from hydraxcel.logging.exception_logging import setup_exception_logging
from hydraxcel.logging.init_logging import (
    LoggingPlatform,
    get_logger,
    init_logging_platform,
)
from hydraxcel.logging.logger_config import create_logging_config
from hydraxcel.logging.mlflow_server import run_mlflow_server

__all__ = [
    "LoggingPlatform",
    "create_logging_config",
    "get_logger",
    "init_logging_platform",
    "log_accelerator_info",
    "log_system_info",
    "run_mlflow_server",
    "setup_exception_logging",
]
