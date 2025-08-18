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
"""Logging helper functions for HydraXcel."""

from pathlib import Path
from typing import Any

__all__ = ["find_project_root", "flatten_dict"]


def find_project_root(current_path: Path, marker: str = "pyproject.toml") -> Path:
    """Traverse up the directory structure to find the root directory.

    Args:
    ----
        current_path (Path): The current file path.
        marker (str): A marker file or directory indicating the root.

    Returns:
    -------
        Path: The root directory path.

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
    """Flatten nested dictionaries for MLflow param logging."""
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
