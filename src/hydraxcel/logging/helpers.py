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
"""Logging helper functions for HydraXcel."""

import logging
from pathlib import Path  # noqa: TC003
from typing import Any

from accelerate import Accelerator

__all__ = [
    "MainProcessFilter",
    "find_project_root",
    "flatten_dict",
]


def find_project_root(current_path: Path, marker: str = "pyproject.toml") -> Path:
    """Traverse up the directory tree to locate the project root.

    Walks parent directories from *current_path* (skipping the filename if it
    ends in ``.py``) until it finds a directory that contains *marker*.

    Args:
        current_path: Starting path for the search (a file or directory).
        marker: Filename or directory name that signals the project root
            (defaults to ``"pyproject.toml"``).

    Returns:
        The first ancestor directory that contains *marker*.

    Raises:
        FileNotFoundError: If *marker* is not found in any parent directory up
            to the filesystem root.

    """
    # Remove the filename if it ends with .py
    current_path = current_path.resolve()
    if current_path.suffix == ".py":
        current_path = current_path.parent

    if (current_path / marker).exists():
        return current_path

    while current_path != current_path.parent:
        # Check if any marker exists in the current directory
        if (current_path / marker).exists():
            return current_path

        # Move one directory up
        current_path = current_path.parent

    # If we reach the filesystem root and haven't found the marker, raise an error
    msg = f"{marker} not found in any parent directories."
    raise FileNotFoundError(msg)


def flatten_dict(
    dictionary: dict[str, Any],
    *,
    parent_key: str = "",
    separator: str = ".",
    max_depth: int | None = 5,
) -> dict[str, Any]:
    """Recursively flatten a nested dictionary into dot-separated keys.

    Joins nested keys with *separator* so that ``{"a": {"b": 1}}`` becomes
    ``{"a.b": 1}``.  Primarily used to prepare Hydra config dicts for MLflow
    ``log_params``, which requires a flat ``{str: str}`` mapping.

    Args:
        dictionary: The (potentially nested) dict to flatten.
        parent_key: Key prefix accumulated through recursion; leave empty for
            top-level calls.
        separator: String inserted between parent and child key segments.
        max_depth: Maximum recursion depth.  ``None`` means unlimited.

    Returns:
        A flat ``dict`` with compound ``separator``-joined keys.

    """
    items: dict[str, Any] = {}
    if max_depth is not None and max_depth < 0:
        return items
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(
                flatten_dict(
                    dictionary=value,
                    parent_key=new_key,
                    separator=separator,
                    max_depth=None if max_depth is None else max_depth - 1,
                ),
            )
        else:
            items[new_key] = value
    return items


class MainProcessFilter(logging.Filter):
    """Pass records only on main (rank 0) process."""

    def __init__(self, name: str = "") -> None:
        """Initialize the logging filter."""
        super().__init__(name)
        self._is_main_process = Accelerator().is_main_process

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: ARG002 # Needed for logging filter function signature
        """Pass records only on main (rank 0) process."""
        try:
            return self._is_main_process
        except (RuntimeError, ValueError) as err:
            logging.getLogger(__name__).warning(
                "Error occurred while checking main process: %s",
                err,
            )
            return False
