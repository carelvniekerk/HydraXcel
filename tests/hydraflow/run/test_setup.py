# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HydraFlow
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
"""Tests for run setup tools."""

from pathlib import Path
from typing import Iterable, Sequence

import pytest
from hydra.conf import RunDir, SweepDir
from hydraflow.run.setup import create_run_dir


def ensure(expr: object, message: str) -> None:
    """Raise AssertionError with message if ``expr`` is falsy."""
    if not expr:
        raise AssertionError(message)


@pytest.mark.parametrize(
    ("config_keys", "expected_segments"),
    [
        ([], ("outputs", "${hydra.job.name}")),
        (
            ["model", "dataset"],
            ("outputs", "${hydra.job.name}", "${model}", "${dataset}"),
        ),
    ],
)
def test_create_run_dir_standard(
    config_keys: list[str],
    expected_segments: Sequence[str],
) -> None:
    """Standard run dir contains expected ordered placeholders and timestamp."""
    result = create_run_dir(root_dir=Path("outputs"), config_keys=config_keys)
    ensure(isinstance(result, RunDir), "Result should be RunDir instance")

    path_parts = Path(result.dir).parts  # type: ignore[arg-type]
    ensure(
        path_parts[: len(expected_segments)] == expected_segments,
        f"Prefix mismatch: {path_parts} vs {expected_segments}",
    )
    ensure(
        path_parts[-1] == "${now:%Y-%m-%d_%H-%M-%S}",
        "Missing/incorrect timestamp placeholder",
    )


@pytest.mark.parametrize(
    ("config_keys", "expected_sub_segments"),
    [
        (["optimizer", "lr"], ("${optimizer}", "${lr}")),
        ([], ()),
    ],
)
def test_create_run_dir_sweep(
    config_keys: list[str],
    expected_sub_segments: Sequence[str],
) -> None:
    """Sweep run dir returns root + subdir with placeholders and timestamp."""
    result = create_run_dir(
        root_dir=Path("multirun"),
        config_keys=config_keys,
        is_sweep=True,
    )
    ensure(isinstance(result, SweepDir), "Result should be SweepDir instance")
    expected_root = str(Path("multirun") / "${hydra.job.name}")
    ensure(result.dir == expected_root, "Sweep root dir mismatch")  # type: ignore[attr-defined]
    sub_parts = Path(result.subdir).parts  # type: ignore[arg-type,attr-defined]
    ensure(
        sub_parts[-1] == "${now:%Y-%m-%d_%H-%M-%S}",
        "Sweep subdir missing timestamp placeholder",
    )
    if expected_sub_segments:
        ensure(
            sub_parts[:-1] == expected_sub_segments,
            "Subdir placeholders mismatch",
        )
    else:
        ensure(len(sub_parts) == 1, "Expected only timestamp when no config keys")


def test_create_run_dir_path_order_is_preserved() -> None:
    """Config key order should be preserved in final path segments."""
    keys = ["alpha", "beta", "gamma"]
    result = create_run_dir(root_dir=Path("outputs"), config_keys=keys)
    parts = Path(result.dir).parts  # type: ignore[arg-type]
    placeholders: Iterable[str] = parts[2:-1]  # skip root + job name + final timestamp
    ensure(
        list(placeholders) == [f"${{{k}}}" for k in keys],
        f"Order not preserved: {placeholders}",
    )
