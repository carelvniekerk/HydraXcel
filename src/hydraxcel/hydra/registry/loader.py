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
"""Loader to scan and import all modules for registry registration."""

import importlib
import pkgutil
from typing import MutableSequence

__all__: list[str] = ["load_methods"]


def load_methods(
    module_name: str,
    module_path: MutableSequence[str],
    excluded_names: list[str] | None = None,
) -> None:
    """Import every sub-module inside a package to trigger registry decoration.

    Iterates over the sub-modules found under *module_path* via
    ``pkgutil.iter_modules`` and imports each one.  Classes decorated with
    ``@registry.register(...)`` inside those modules execute their registration
    side-effect on import, so this function is the standard way to ensure all
    variants in a plugin directory are available in the Hydra config store.

    Args:
        module_name: Fully-qualified package name (e.g.
            ``"myproject.models"``).
        module_path: The ``__path__`` of the package to scan.
        excluded_names: Sub-module names to skip during the scan.

    """
    if excluded_names is None:
        excluded_names = []
    for _, method_name, _ in pkgutil.iter_modules(module_path):
        if method_name in excluded_names:
            continue
        method_module = f"{module_name}.{method_name}"
        importlib.import_module(method_module)
