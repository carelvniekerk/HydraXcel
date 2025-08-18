# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HydraXcel
# Author: Carel van Niekerk, Benjamin Ruppik
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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Initialise the weights and biases logging."""

from pathlib import Path

import mlflow
from omegaconf import DictConfig, OmegaConf

from hydraxcel.logging.helpers import find_project_root, flatten_dict

__all__ = ["initialize_mlflow"]


def initialize_mlflow(  # noqa: PLR0913
    config: DictConfig | None = None,
    experiment_name: str = "ConfidentLLM",
    tracking_subdir: str = ".mlflow",
    run_name: str | None = None,
    *,
    nested: bool = False,
    max_value_len: int = 250,
) -> None:
    """Initialize MLflow tracking.

    Args:
        config: Hydra/OmegaConf configuration to log.
        experiment_name: MLflow experiment name.
        tracking_subdir: Directory (under project root) for file store.
        run_name: Optional MLflow run name.
        nested: Start a nested run if already inside a parent run.
        max_value_len: Maximum length for parameter values.

    Returns:
        The active MLflow run.

    """
    project_root: Path = find_project_root(Path(__file__))
    tracking_dir: Path = project_root / tracking_subdir
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(experiment_name)

    active_run: mlflow.ActiveRun | None = mlflow.active_run()  # type: ignore[possibly-unbound-attr] # Ty Bug
    if active_run is None or nested:
        mlflow.start_run(run_name=run_name, nested=nested)  # type: ignore[possibly-unbound-attr] # Ty Bug

    cfg_container = OmegaConf.to_container(config, resolve=True)
    if isinstance(cfg_container, dict):
        flat_params = flatten_dict(cfg_container)
        # Convert values to strings (MLflow params must be str-able & short)
        safe_params: dict[str, str] = {}
        for key, value in flat_params.items():
            try:
                str_value = str(value)
                if len(str_value) > max_value_len:
                    str_value = str_value[: max_value_len - 3] + "..."
                safe_params[key] = str_value
            except TypeError:
                continue
        # Chunk if too many params (MLflow can handle many, but keep reasonable)
        mlflow.log_params(safe_params)  # type: ignore[possibly-unbound-attr] # Ty Bug

        # Also log full config as an artifact (YAML)
        yaml_txt = OmegaConf.to_yaml(config)
        mlflow.log_text(yaml_txt, artifact_file="config.yaml")  # type: ignore[possibly-unbound-attr] # Ty Bug
    else:
        mlflow.log_text(str(cfg_container), artifact_file="config_repr.txt")  # type: ignore[possibly-unbound-attr] # Ty Bug
