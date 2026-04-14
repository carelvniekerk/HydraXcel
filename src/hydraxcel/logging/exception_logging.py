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
"""Set up exception logging."""

import logging
import os
import sys
import warnings

logger = logging.getLogger("__main__")

__all__ = ["setup_exception_logging"]


def setup_exception_logging(
    logger: logging.Logger = logger,
) -> None:
    """Set up a custom exception handler that logs uncaught exceptions.

    Enables full Hydra tracebacks (``HYDRA_FULL_ERROR=1``), replaces
    ``sys.excepthook`` with a handler that logs critical-level messages for
    unhandled exceptions (except ``KeyboardInterrupt``), and redirects Python
    warnings through the same logger.

    Args:
        logger: Logger instance to which uncaught exceptions and warnings are
            written.  Defaults to the module-level ``__main__`` logger.

    """
    # Setting this environment variable to "1" makes Hydra print the full stack trace.
    print("Setting HYDRA_FULL_ERROR environment variable to '1'.")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(f"{os.environ['HYDRA_FULL_ERROR'] = }")

    def handle_exception(
        exc_type,  # noqa: ANN001
        exc_value,  # noqa: ANN001
        exc_traceback,  # noqa: ANN001
    ) -> None:
        """Log uncaught exceptions at CRITICAL level, except KeyboardInterrupt.

        Assigned to ``sys.excepthook``; do not call directly.
        ``KeyboardInterrupt`` is forwarded to the default hook so that Ctrl-C
        terminates the process cleanly.

        Args:
            exc_type: The exception class.
            exc_value: The exception instance.
            exc_traceback: The associated traceback object.

        """
        if issubclass(
            exc_type,
            KeyboardInterrupt,
        ):
            sys.__excepthook__(
                exc_type,
                exc_value,
                exc_traceback,
            )
            return

        logger.critical(
            "Uncaught exception",
            exc_info=(
                exc_type,
                exc_value,
                exc_traceback,
            ),
        )

    def handle_warning(  # noqa: PLR0913
        message,  # noqa: ANN001
        category,  # noqa: ANN001
        filename,  # noqa: ANN001
        lineno,  # noqa: ANN001
        file=None,  # noqa: ANN001, ARG001
        line=None,  # noqa: ANN001, ARG001
    ) -> None:
        """Log a Python warning through the configured logger.

        Signature matches ``warnings.showwarning`` so it can be assigned to
        ``warnings.showwarning`` directly.

        Args:
            message: The warning message object.
            category: The warning category class (e.g. ``DeprecationWarning``).
            filename: Source file where the warning was issued.
            lineno: Line number in *filename*.
            file: Unused (present for ``warnings.showwarning`` compatibility).
            line: Unused (present for ``warnings.showwarning`` compatibility).

        """
        logger.warning(f"{category.__name__} at {filename}:{lineno}: {message}")  # noqa: G004

    sys.excepthook = handle_exception
    warnings.showwarning = handle_warning  # ty:ignore[invalid-assignment]
