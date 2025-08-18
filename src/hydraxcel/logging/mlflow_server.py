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
"""Host the MLFlow server."""

import sys
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

from hydra import main
from hydra.core.config_store import ConfigStore

from mlflow.server import _run_server
from mlflow.server.handlers import initialize_backend_stores
from mlflow.utils.process import ShellCommandException

logger = getLogger("mlflow")


@dataclass
class ServerConfig:
    """MLFlow server configuration."""

    backend_store_uri: Path = Path("mlflow")
    host: str = "127.0.0.1"
    port: int = 5000
    dev: bool = False


ConfigStore.instance().store(
    name="server_config",
    node=ServerConfig,
)


@main(config_path=None, config_name="server_config", version_base="1.3")
def run_mlflow_server(cfg: ServerConfig) -> None:
    """Run the MLFlow server."""
    try:
        initialize_backend_stores(
            backend_store_uri=cfg.backend_store_uri.as_posix(),
            registry_store_uri=None,
            default_artifact_root=None,
        )
    except Exception as exception:
        logger.error("Error initializing backend store")  # noqa: TRY400
        logger.exception(exception)  # noqa: TRY401
        sys.exit(1)

    try:
        _run_server(
            file_store_path=cfg.backend_store_uri.as_posix(),
            registry_store_uri=cfg.backend_store_uri.as_posix(),
            default_artifact_root=cfg.backend_store_uri.as_posix(),
            serve_artifacts=True,
            artifacts_only=False,
            artifacts_destination=None,
            host=cfg.host,
            port=cfg.port,
            workers=4,
        )
    except ShellCommandException:
        logger.error(  # noqa: TRY400
            "Running the mlflow server failed. Please see the logs above for details.",
        )
        sys.exit(1)


if __name__ == "__main__":
    run_mlflow_server()
