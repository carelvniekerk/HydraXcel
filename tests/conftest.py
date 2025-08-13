# coding=utf-8  # noqa: INP001
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
"""Fixtures for testing."""

import os
import sys
from pathlib import Path
from typing import Callable, Generator

import pytest
from accelerate import Accelerator
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

CONSTANT: int = 42


@pytest.fixture(autouse=True)
def hydra_reset() -> Generator[None]:
    """Reset Hydra state before each test."""
    # Make sure each test starts with a clean Hydra state
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


@pytest.fixture(autouse=True)
def clean_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove pytest's CLI args so hydra.main() doesn't choke on them."""
    monkeypatch.setattr(sys, "argv", ["pytest_hydra_test"])


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test in its own temp directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def job_name() -> str:
    """Derive job name from the test file name."""
    return Path(__file__).stem


@pytest.fixture
def hydra_config_dir(tmp_path: Path, job_name: str) -> Path:
    """Create a config directory containing the config file named after job_name."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    job_config_file: Path = cfg_dir / f"{job_name}.yaml"
    job_config_file.write_text(f"defaults:\n  - _self_\n\nconstant: {CONSTANT}\n")
    return cfg_dir


@pytest.fixture
def wandb_init(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Mock W&B initialization call."""
    calls: dict[str, str] = {}

    def fake_initialize_wandb(*, config: DictConfig, project_name: str) -> None:
        calls["project_name"] = project_name
        calls["constant"] = getattr(config, "constant", None)

    # Patch the symbol imported into hydraflow.run.setup
    monkeypatch.setattr(
        "hydraflow.run.setup.initialize_wandb",
        fake_initialize_wandb,
        raising=True,
    )
    return calls


@pytest.fixture
def disable_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable debug mode for tests."""
    if "ACCELERATE_DEBUG_MODE" in os.environ:
        monkeypatch.delenv("ACCELERATE_DEBUG_MODE", raising=False)


@pytest.fixture
def enable_debug(monkeypatch: pytest.MonkeyPatch) -> str:
    """Enable debug mode for tests."""
    monkeypatch.setenv("ACCELERATE_DEBUG_MODE", "1")
    return "1"


@pytest.fixture
def make_user_main() -> Callable[
    [dict[str, str] | None],
    Callable[[DictConfig, Accelerator], None],
]:
    """Build a user_main with optional recording / extra assertions."""

    def _make(
        *,
        record: dict[str, str] | None = None,
    ) -> Callable[[DictConfig, Accelerator], None]:
        def _user_main(cfg: DictConfig, accelerator: Accelerator) -> None:
            # Core assertion shared by tests
            if cfg.constant != CONSTANT:
                raise AssertionError(f"Expected {CONSTANT}, got {cfg.constant}.")  # noqa: EM102, TRY003
            # Optional recording
            if record is not None:
                record["device"] = str(accelerator.device)
                record["constant"] = cfg.constant
                record["working_dir"] = str(Path.cwd())

        return _user_main

    return _make  # type: ignore[invalid-return-type] # Remove once Todo in Ty is solved


@pytest.fixture
def config_constant() -> int:
    """Provide the constant value for configuration."""
    return CONSTANT
