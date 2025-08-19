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
"""Tests for the launch command."""

import sys
from pathlib import Path
from typing import Callable

import pytest

from hydraxcel import launch


def ensure(expr: object, message: str) -> None:
    """Raise AssertionError with message if ``expr`` is falsy."""
    if not expr:
        raise AssertionError(message)


def test_launch_injects_script_and_args_no_delimiter(
    monkeypatch: pytest.MonkeyPatch,
    accelerate_config_dir: Path,
    dummy_script: Path,
    patch_launch_command: Callable[[dict[str, str | list[str]] | None], None],
) -> None:
    """Test launch command with script and args containing no delimiter."""
    # Simulate CLI: all args become passthrough (no '--' present)
    monkeypatch.setattr(sys, "argv", ["prog", "--foo", "bar", "123"])

    captured: dict[str, str | list[str]] = {}
    patch_launch_command(record=captured)  # type: ignore[missing-argument]

    launcher = launch(
        script_path=dummy_script,
        hydra_configs_dir=str(accelerate_config_dir),
    )
    launcher()  # type: ignore[missing-argument]

    ensure(
        Path(captured["training_script"]) == dummy_script,
        "Training script path mismatch",
    )
    ensure(
        captured["training_script_args"] == ["--foo", "bar", "123"],
        "Training script args mismatch",
    )
    # sys.argv should have been trimmed for Hydra before launch() decoration
    ensure(
        sys.argv == ["prog"],
        "sys.argv was not trimmed correctly",
    )


def test_launch_injects_script_and_args_with_delimiter(
    monkeypatch: pytest.MonkeyPatch,
    accelerate_config_dir: Path,
    dummy_script: Path,
    patch_launch_command: Callable[[dict[str, str | list[str]] | None], None],
) -> None:
    """Test launch command with script and args containing -- delimiter."""
    # Args before '--' => passthrough to training_script_args
    # Args after '--' remain for Hydra (we keep one harmless override style arg)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--lr", "3e-4", "--batch", "8", "--", "cpu=True"],
    )

    captured: dict[str, str | list[str]] = {}
    patch_launch_command(record=captured)  # type: ignore[missing-argument]

    launcher = launch(
        script_path=dummy_script,
        hydra_configs_dir=str(accelerate_config_dir),
    )
    launcher()  # type: ignore[missing-argument]

    ensure(
        Path(captured["training_script"]) == dummy_script,
        "Training script path mismatch",
    )
    # Only the args before the '--' delimiter
    ensure(
        captured["training_script_args"] == ["--lr", "3e-4", "--batch", "8"],
        "Training script args mismatch",
    )
    # sys.argv after extraction keeps items after '--' (Hydra sees them)
    ensure(
        sys.argv == ["prog", "cpu=True"],
        "sys.argv was not trimmed correctly",
    )
