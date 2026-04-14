# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HydraXcel
# Author: Carel van Niekerk, Benjamin Ruppik
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
"""Run entry-point utilities for HydraXcel.

Provides seed initialisation, Hydra/logging setup helpers, and the
``hydraxcel_main`` decorator that wires a user-supplied main function to
Accelerate, Hydra config management, and an experiment-tracking platform.
"""

import random
from dataclasses import is_dataclass
from functools import wraps
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from accelerate import Accelerator
from hydra import main
from hydra.conf import HydraConf, JobConf, RunDir, SweepDir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from hydraxcel.logging import (
    LoggingPlatform,
    create_logging_config,
    get_logger,
    init_logging_platform,
    log_accelerator_info,
    log_system_info,
    setup_exception_logging,
)

__all__ = [
    "_setup_hydra_config_and_logging",
    "hydraxcel_main",
    "set_seed",
]
logger = get_logger("__main__")


def _create_run_dir(
    root_dir: Path,
    config_keys: list[str],
    *,
    is_sweep: bool = False,
) -> RunDir | SweepDir:
    """Build a Hydra run or sweep directory spec from config keys.

    Constructs a ``RunDir`` or ``SweepDir`` whose path components are Hydra
    interpolation placeholders derived from the supplied config keys, followed
    by a timestamp segment.

    Args:
        root_dir: Base directory under which outputs are written (e.g.
            ``Path("outputs")`` or ``Path("multirun")``).
        config_keys: Ordered list of dot-separated config attribute names whose
            runtime values are embedded as path segments.
        is_sweep: When ``True`` returns a ``SweepDir`` (multirun layout) with
            key placeholders in the *subdir*; otherwise returns a ``RunDir``
            with placeholders in the *dir* path.

    Returns:
        A ``RunDir`` for single runs or a ``SweepDir`` for multirun/sweep
        launches, each containing Hydra interpolation strings.

    """
    run_dir: Path = root_dir / "${hydra.job.name}"

    if is_sweep:
        sub_dir: Path | None = (
            Path("${" + config_keys[0] + "}") if config_keys else None
        )
        for key in config_keys[1:]:
            _key = "${" + key + "}"
            sub_dir = sub_dir / _key  # ty:ignore[unsupported-operator]

        sub_dir = (
            sub_dir / "${now:%Y-%m-%d_%H-%M-%S}"
            if sub_dir
            else Path("${now:%Y-%m-%d_%H-%M-%S}")
        )

        return SweepDir(
            dir=str(run_dir),
            subdir=str(sub_dir),
        )

    for key in config_keys:
        _key = "${" + key + "}"
        run_dir = run_dir / _key

    run_dir = run_dir / "${now:%Y-%m-%d_%H-%M-%S}"

    return RunDir(str(run_dir))


def _setup_hydra_config_and_logging(
    *,
    file_path: Path = Path(__file__),
    config_keys: list[str],
    change_to_output_dir: bool = True,
    add_submission_launcher: bool = False,
) -> str:
    """Register Hydra run/sweep directories and job-logging config in the config store.

    Installs a ``HydraConf`` node into the global ``ConfigStore`` so that any
    subsequent ``@hydra.main`` call picks up the correct output paths and log
    formatting.  Optionally activates the ``job_submission`` launcher for
    cluster-based multirun sweeps.

    Args:
        file_path: Path to the calling script; its stem is used as the Hydra
            job name.
        config_keys: Ordered list of config attribute names used to construct
            output directory path segments.
        change_to_output_dir: When ``True`` (default), Hydra changes the
            working directory to the run output folder.
        add_submission_launcher: When ``True``, adds the ``job_submission``
            launcher to Hydra defaults for cluster job submission.

    Returns:
        The derived job name (stem of *file_path*).

    """
    job_name: str = file_path.stem
    setup_exception_logging(logger)

    job_config: JobConf = JobConf(name=job_name, chdir=change_to_output_dir)
    logging_config: dict = create_logging_config()

    run_dir: RunDir = _create_run_dir(
        root_dir=Path("outputs"),
        config_keys=config_keys,
    )  # ty:ignore[invalid-assignment]
    sweep_dir: SweepDir = _create_run_dir(
        root_dir=Path("multirun"),
        config_keys=config_keys,
        is_sweep=True,
    )  # ty:ignore[invalid-assignment]

    if add_submission_launcher:
        hydra_defaults: list[str | dict[str, str | None]] = [
            # Standard defaults
            "_self_",
            {"sweeper": "basic"},
            {"help": "default"},
            {"hydra_help": "default"},
            {"hydra_logging": "default"},
            {"callbacks": None},
            # Set launcher
            {"launcher": "job_submission"},
        ]

        hydra_config: HydraConf = HydraConf(
            defaults=hydra_defaults,
            job=job_config,
            job_logging=logging_config,
            run=run_dir,
            sweep=sweep_dir,
        )
    else:
        hydra_config = HydraConf(
            job=job_config,
            job_logging=logging_config,
            run=run_dir,
            sweep=sweep_dir,
        )

    store = ConfigStore.instance().store
    store(
        node=hydra_config,
        name="config",
        group="hydra",
    )

    return job_name


def _get_cfg_attr(cfg: DictConfig, attr: str) -> str | float | bool:
    if "." in attr:
        main_key, sub_key = attr.split(".", 1)
        return _get_cfg_attr(getattr(cfg, main_key), sub_key)
    return getattr(cfg, attr)


def get_job_name(cfg: DictConfig, keys: list[str]) -> str:
    """Get the job name from the configuration."""
    key_values = [_get_cfg_attr(cfg, key) for key in keys]
    return "_".join(str(value) for value in key_values)


def set_seed(seed: int) -> None:
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.default_rng(seed)


def hydraxcel_main(  # noqa: PLR0913
    project_name: str,
    *,
    config_class: type | None = None,
    output_dir_keys: list[str] | None = None,
    hydra_configs_dir: str | None = None,
    hydra_base_version: str = "1.3",
    logging_platform: LoggingPlatform | str = LoggingPlatform.WANDB,
    job_name_keys: list[str] | None = None,
    add_hydra_submission_launcher: bool = False,
) -> Callable[Callable[..., None], Callable[..., None]]:
    """Wire a training function to Hydra, Accelerate, and an experiment tracker.

    Wraps a user-defined ``main(cfg, accelerator)`` function so that it is
    invoked through ``hydra.main``, receives an ``Accelerator`` instance, and
    initialises the chosen experiment-tracking platform before calling into
    user code.

    Args:
        project_name: Top-level project/experiment name passed to the logging
            platform (e.g. W&B project or MLflow experiment).
        config_class: Optional structured-config dataclass whose fields define
            the Hydra schema.  Mutually exclusive with *hydra_configs_dir*.
        output_dir_keys: Ordered list of config attribute names embedded as
            path segments in the Hydra output directory.
        hydra_configs_dir: Path to a directory of YAML config files.  Mutually
            exclusive with *config_class*.
        hydra_base_version: Hydra ``version_base`` string (default ``"1.3"``).
        logging_platform: Experiment-tracking backend to initialise.  Accepts
            a ``LoggingPlatform`` enum value or its string name.
        job_name_keys: Config attribute names whose values are joined with
            underscores to form the run/job name reported to the tracking
            platform.
        add_hydra_submission_launcher: When ``True``, registers the
            ``job_submission`` Hydra launcher for cluster-based sweeps.

    Returns:
        A decorator that accepts the user main function and returns a
        Hydra-compatible entry-point callable.

    Raises:
        ValueError: If both *config_class* and *hydra_configs_dir* are
            provided, or if *config_class* is not a dataclass.

    """
    if not isinstance(logging_platform, LoggingPlatform):
        logging_platform: LoggingPlatform = LoggingPlatform(logging_platform)

    if output_dir_keys is None:
        output_dir_keys: list[str] = []

    if config_class is not None and hydra_configs_dir is not None:
        raise ValueError(  # noqa: TRY003
            "If config_class is provided, hydra_configs_dir cannot be set.",  # noqa: EM101
        )

    if config_class is not None and not is_dataclass(config_class):
        raise ValueError("If config_class is provided, it must be a dataclass.")  # noqa: EM101, TRY003

    def outer(main_func: Callable[..., None]) -> Callable[..., None]:
        """Run the main function with the Accelerator."""
        task_name = _setup_hydra_config_and_logging(
            file_path=Path(main_func.__code__.co_filename),  # ty:ignore[unresolved-attribute]
            config_keys=output_dir_keys,
            add_submission_launcher=add_hydra_submission_launcher,
        )

        if config_class is not None:
            hydra_store = ConfigStore.instance().store
            hydra_store(
                node=config_class,
                name=task_name,
            )

        @wraps(main_func)
        @main(
            version_base=hydra_base_version,
            config_path=hydra_configs_dir,
            config_name=task_name,
        )
        def acc_main_func(cfg: DictConfig) -> None:
            log_system_info()
            accelerator: Accelerator = Accelerator()
            log_accelerator_info(accelerator)
            job_name: str | None = (
                get_job_name(
                    cfg=cfg,
                    keys=job_name_keys,
                )
                if job_name_keys
                else None
            )
            init_logging_platform(
                platform=logging_platform,
                config=cfg,
                project_name=project_name,
                task_name=task_name,
                job_name=job_name,
                accelerator=accelerator,
            )
            try:
                main_func(cfg, accelerator)
            finally:
                # Do not manually end WANDB run
                accelerator.trackers = (
                    []
                    if logging_platform == LoggingPlatform.WANDB
                    else accelerator.trackers
                )
                accelerator.end_training()

        return acc_main_func

    return outer
