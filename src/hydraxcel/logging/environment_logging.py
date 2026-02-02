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
"""Logging configuration for the HydraXcel project."""

import logging
import os
import platform
import socket
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from git import Repo

__all__ = [
    "log_accelerator_info",
    "log_system_info",
]


def log_system_info() -> None:
    """Log system hostname and environment information."""
    try:
        hostname = socket.gethostname()
    except Exception:  # noqa: BLE001 - We want to proceed no matter what the error is
        hostname = "sys"

    logger: logging.Logger = logging.getLogger(hostname)

    logger.info(
        msg=f"Running on {hostname = }",  # noqa: G004 - low overhead
    )

    _log_python_env_info(logger)
    _log_git_info()


def _log_git_info() -> None:
    """Get the git info of the current branch and commit hash."""
    logger = logging.getLogger("git")
    try:
        repo = Repo(
            path=Path(__file__).resolve().parent,
            search_parent_directories=True,
        )
        branch_name: str = repo.active_branch.name
        commit_hex: str = repo.head.object.hexsha
        logger.info("Version Control")
        logger.info(
            msg=f"\tBranch Name:\t{branch_name}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"\tCommit Hex:\t{commit_hex}",  # noqa: G004 - low overhead
        )
    except Exception:  # noqa: BLE001 - We want to proceed no matter what the error is
        logger.info(
            msg="Unable to determine git branch/commit",
        )


def _log_python_env_info(logger: logging.Logger) -> None:
    """Log Python environment and Poetry information."""
    # Log Python version and executable
    try:
        python_version: str = sys.version.split()[0]
        python_path: str = sys.executable
        logger.info(
            msg=f"Python version: {python_version}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Python executable: {python_path}",  # noqa: G004 - low overhead
        )
    except Exception:  # noqa: BLE001 - We want to proceed no matter what the error is
        logger.info(msg="Unable to determine Python version/path")

    try:
        # Virtualenv Section
        python_version = sys.version.split()[0]
        implementation = platform.python_implementation()
        executable = Path(sys.executable)
        venv_path = executable.parent.parent if "VIRTUAL_ENV" in os.environ else None
        valid_venv = bool(venv_path and (venv_path / "bin" / "python").exists())

        logger.info("Virtualenv")
        logger.info(f"\tPython:\t\t{python_version}")  # noqa: G004
        logger.info(f"\tImplementation:\t{implementation}")  # noqa: G004
        logger.info(
            f"\tPath:\t\t{venv_path if venv_path else 'Not in a virtual environment'}",  # noqa: G004
        )
        logger.info(f"\tExecutable:\t{executable}")  # noqa: G004
        logger.info(f"\tValid:\t\t{valid_venv}")  # noqa: G004

        # Base Section
        base_path = Path(sys.base_prefix)
        platform_name = platform.system().lower()
        os_type = os.name
        base_executable = base_path / "bin" / f"python{python_version[:3]}"
        if not base_executable.exists():
            base_executable = Path(sys.base_exec_prefix)

        logger.info("Base")
        logger.info(f"\tPlatform:\t{platform_name}")  # noqa: G004
        logger.info(f"\tOS:\t\t{os_type}")  # noqa: G004
        logger.info(f"\tPython:\t\t{python_version}")  # noqa: G004
        logger.info(f"\tPath:\t\t{base_path}")  # noqa: G004
        logger.info(f"\tExecutable:\t{base_executable}")  # noqa: G004

    except Exception:
        logger.exception("Error logging environment info.")


def log_accelerator_info(accelerator: Accelerator) -> None:
    """Log information about the Accelerator."""
    logger = logging.getLogger("accelerate")
    logger.info("Accelerator Information:")
    logger.info(f"\tDevice:\t\t\t{accelerator.device}")  # noqa: G004
    logger.info(f"\tDistributed Type:\t{accelerator.distributed_type}")  # noqa: G004
    logger.info(f"\tNum Processes:\t\t{accelerator.num_processes}")  # noqa: G004
    logger.info(f"\tLocal Process Index:\t{accelerator.local_process_index}")  # noqa: G004
    logger.info(f"\tMain Process:\t\t{accelerator.is_main_process}")  # noqa: G004
    logger.info(f"\tMixed Precision:\t{accelerator.mixed_precision}")  # noqa: G004
    if getattr(accelerator.state, "gradient_accumulation_steps", None):
        logger.info(
            f"\tGrad Accumulation:\t{accelerator.state.gradient_accumulation_steps}",  # noqa: G004
        )
    logger.info(
        f"\tCUDA_VISIBLE_DEVICES:\t{os.getenv('CUDA_VISIBLE_DEVICES', 'None')}",  # noqa: G004
    )
    logger.info(
        f"\tDEBUG:\t\t\t{os.getenv('ACCELERATE_DEBUG_MODE', 'false')}",  # noqa: G004
    )
    logger.info(
        f"\tACCELERATE_LOG_LEVEL:\t{os.getenv('ACCELERATE_LOG_LEVEL', 'None')}",  # noqa: G004
    )
    if torch.cuda.is_available():
        logger.info("CUDA")
        logger.info(f"\tDevice Count:\t{torch.cuda.device_count()}")  # noqa: G004
        logger.info(f"\tCurrent Device:\t{torch.cuda.current_device()}")  # noqa: G004
        logger.info(
            f"\tDevice Name:\t{torch.cuda.get_device_name(torch.cuda.current_device())}"  # noqa: COM812, G004
        )
    if torch.mps.is_available():
        logger.info("MPS")
        logger.info(f"\tDevice Count:\t{torch.mps.device_count()}")  # noqa: G004
