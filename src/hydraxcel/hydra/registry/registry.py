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
"""HydraXcel Hydra Registry definition."""

from collections.abc import Callable

from hydra.core.config_store import ConfigStore

type RegistryDecorator[RegistryItemT] = Callable[[RegistryItemT], RegistryItemT]

__all__ = ["BaseRegistry"]

config_store = ConfigStore.instance()


class BaseRegistry[RegistryItemT]:
    """Base class for registries that hold typed items."""

    def __init__(self, group_name: str) -> None:
        """Initialize the registry."""
        self._registry: dict[str, RegistryItemT] = {}
        self._group_name = group_name

    def register(
        self,
        name: str,
        **kwargs: object,  # noqa: ARG002
    ) -> RegistryDecorator[RegistryItemT]:
        """Register an item with a given name."""

        def decorator(cls: RegistryItemT) -> RegistryItemT:
            self._registry[name] = cls

            config_store.store(
                name=name,
                group=self._group_name,
                node=cls,
            )
            return cls

        return decorator

    def get(self, name: str) -> RegistryItemT:
        """Retrieve an item by its name."""
        if name not in self._registry:
            msg = f"Item '{name}' not found in registry group '{self._group_name}'."
            raise KeyError(msg)
        return self._registry.get(name)  # type: ignore[invalid-return-type]
