# AGENTS.md — HydraXcel

> Configuration and context for AI coding agents working on HydraXcel

## Project Overview

**HydraXcel** is a configuration-driven deep learning experiment launcher that combines:
- **Hydra** (Facebook): Configuration management framework
- **Accelerate** (HuggingFace): Distributed/multi-GPU training toolkit
- **UV**: Fast Python package manager and workflow tool

The project simplifies running reproducible ML experiments by providing declarative configuration (via YAML or Python dataclasses), automatic experiment logging (MLflow/W&B), and seamless distributed training support.

**Version**: 0.2.0
**Python**: >=3.12 (targets 3.13)
**License**: Apache 2.0

## Project Structure

```
HydraXcel/
├── src/hydraxcel/                    # Main package
│   ├── __init__.py                   # Public API exports
│   ├── run/                          # Experiment execution
│   │   ├── __init__.py
│   │   └── setup.py                  # Core decorator (@hydraxcel_main) and run setup
│   ├── logging/                      # Logging and experiment tracking
│   │   ├── __init__.py
│   │   ├── init_logging.py           # Platform initialization (LOCAL/WANDB/MLFLOW)
│   │   ├── init_mlflow.py            # MLflow integration
│   │   ├── init_wandb.py             # Weights & Biases integration
│   │   ├── logger_config.py          # Logger configuration
│   │   ├── mlflow_server.py          # MLflow server runner (entry point)
│   │   ├── environment_logging.py    # System/environment info logging
│   │   ├── exception_logging.py      # Exception handling
│   │   └── helpers.py                # Logging utilities
│   ├── accelerate/                   # Distributed training launcher
│   │   ├── __init__.py
│   │   ├── launch_tools.py           # Main launch() function implementation
│   │   └── config.py                 # LaunchConfig dataclass
│   ├── hydra/                        # Hydra extensions and utilities
│   │   ├── __init__.py
│   │   ├── configuration.py          # flatten_config, hydra_config utilities
│   │   ├── registration.py           # Plugin registration
│   │   └── registry/                 # Generic registry pattern
│   │       ├── __init__.py
│   │       ├── registry.py           # BaseRegistry class
│   │       └── loader.py             # Module loader for registrations
│   └── resolvers/                    # Custom Hydra/OmegaConf resolvers
│       ├── __init__.py
│       └── class_name.py             # class_name resolver
├── tests/                            # Test suite
│   ├── conftest.py                   # Pytest fixtures
│   └── hydraxcel/
│       ├── accelerate/
│       │   └── test_launch_tools.py
│       └── run/
│           └── test_setup.py
├── examples/                         # Usage examples
│   ├── __init__.py
│   ├── hello.py                      # Simple example script
│   └── configs/                      # Example Hydra configurations
├── pyproject.toml                    # Project metadata and dependencies
├── .pre-commit-config.yaml           # Pre-commit hooks configuration
└── README.md                         # Main documentation
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `run/setup.py` | Core `@hydraxcel_main` decorator, seed management, output directory creation |
| `logging/init_logging.py` | Logging platform initialization (LOCAL/WANDB/MLFLOW) |
| `accelerate/launch_tools.py` | `launch()` function for distributed training |
| `accelerate/config.py` | `LaunchConfig` dataclass with Accelerate options |
| `hydra/configuration.py` | `flatten_config()`, `hydra_config()` utilities |
| `hydra/registry/` | Generic registry pattern for extensibility |

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests with coverage
uv run pytest

# Lint and format
uv run ruff check --fix .
uv run ruff format .

# Type checking
uv run ty check

# Run pre-commit hooks manually
pre-commit run --all-files

# Run a specific test file
uv run pytest tests/hydraxcel/run/test_setup.py -v
```

## Testing

- **Framework**: Pytest with coverage (`pytest-cov`)
- **Coverage target**: `src/hydraxcel`
- **Key fixtures** (in `tests/conftest.py`):
  - `hydra_reset`: Clears Hydra state between tests (autouse)
  - `clean_sys_argv`: Removes pytest args to avoid Hydra conflicts (autouse)
  - `isolated_cwd`: Runs test in temp directory
  - `patch_launch_command`: Monkeypatches Accelerate launcher
  - `logging_platform_init`: Mocks logging platform calls

Run tests before committing - pre-commit hooks enforce this.

## Dependencies

### Core
- `accelerate>=1.10.0` - HuggingFace distributed training
- `hydra-core` - Configuration management (from GitHub)
- `mlflow>=3.2.0` - Experiment tracking
- `wandb>=0.21.1` - Weights & Biases integration
- `transformers>=4.55.0` - NLP models (logging utilities)
- `colorlog>=6.9.0` - Colored console logging

### Development
- `pytest-cov>=6.2.1` - Testing with coverage
- Ruff - Linting and formatting
- `ty` - Type checking (Astral)
- Bandit - Security scanning

## Code Conventions

### Style
- **Docstrings**: Google-style with `Args`, `Returns`, `Raises` sections
- **Strings**: Double quotes (`"`)
- **Type unions**: Use `|` syntax (`str | None`), not `Optional`
- **Type aliases**: Use `TypeAlias` from `typing`
- **Exceptions**: Raise specific exceptions (`ValueError`, `TypeError`)

### Patterns

**Dataclass configs** for structured configuration:
```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    type: str = "ResNet50"
    pretrained: bool = True
```

**Decorator pattern** for entry points:
```python
@hydraxcel_main(
    "ProjectName",
    config_class=TrainConfig,
    logging_platform=LoggingPlatform.WANDB,
)
def main(cfg):
    ...
```

**Registry pattern** for extensibility:
```python
@registry.register("component_name")
def my_component():
    ...
```

### Linting Rules

Ruff is configured with `select = ["ALL"]` and specific ignores. Key rules:
- No trailing whitespace
- Consistent imports
- Type annotations required
- Tests exempt from `INP001` (implicit namespace packages)

## Architecture Notes

### Design Principles
1. **Configuration as Code**: Configs can be Python dataclasses or YAML
2. **Decorator-Driven**: Minimal boilerplate through decorators
3. **Single-Device Abstraction**: User code runs on single device, Accelerate handles distribution
4. **Structured Outputs**: Automatic organization with timestamps and parameters
5. **Multi-Platform Logging**: Flexible backends (local/WANDB/MLflow)

### Key Abstractions

**`@hydraxcel_main`** wraps user functions to:
- Initialize Hydra with config_class or YAML configs
- Set up logging (colorlog with Hydra integration)
- Route to logging platforms (LOCAL/WANDB/MLFLOW)
- Create structured output directories: `outputs/<job_name>/<param_keys>/<timestamp>`
- Handle exceptions with traceback logging

**`launch()`** creates distributed training launchers:
- Takes script path, returns callable
- Bridges Hydra config with Accelerate launch_command
- Supports passthrough arguments via `--`

**`LaunchConfig`** is a frozen dataclass with:
- Accelerate options (FP precision, backends, FSDP strategies)
- Validation for allowed option values

### Output Directory Structure
- Standard runs: `outputs/<job_name>/<param_keys>/<timestamp>`
- Sweep runs: `multirun/<job_name>/<param_keys>/<timestamp>`

## Common Tasks

### Adding a New Logging Platform
1. Create `init_<platform>.py` in `src/hydraxcel/logging/`
2. Add platform to `LoggingPlatform` enum in `init_logging.py`
3. Add initialization case in `init_logging_platform()`
4. Add tests in `tests/hydraxcel/logging/`

### Adding a New Accelerate Option
1. Add field to `LaunchConfig` dataclass in `accelerate/config.py`
2. Update validation if needed (allowed values)
3. Update `launch_tools.py` to pass the option to Accelerate
4. Add tests in `tests/hydraxcel/accelerate/`

### Adding a New OmegaConf Resolver
1. Create resolver function in `src/hydraxcel/resolvers/`
2. Register with OmegaConf in `__init__.py`
3. Document usage pattern

## Git Workflow

- **Main branch**: `main`
- **Commit format**: `<type>: <description>`
  - `feat:` - New feature
  - `fix:` - Bug fix
  - `refactor:` - Code restructuring
  - `docs:` - Documentation
  - `test:` - Tests
  - `chore:` - Maintenance

Pre-commit hooks run automatically:
- Ruff lint + format
- UV lock sync
- Pytest
- Type checking with `ty`

## Gotchas and Tips

1. **Hydra state**: Tests must reset Hydra state between runs (handled by `hydra_reset` fixture)
2. **sys.argv conflicts**: Pytest args can conflict with Hydra (handled by `clean_sys_argv` fixture)
3. **DeepSpeed**: Optional dependency - code should gracefully handle when not installed
4. **Version mismatch**: README says 0.1.0a but pyproject.toml has 0.2.0 - use pyproject.toml as source of truth
5. **hydra-core**: Installed from GitHub, not PyPI - ensure `uv.lock` is in sync

## Entry Points

Defined in `pyproject.toml`:
```toml
[project.scripts]
tests = "pytest:main"
```

User projects typically add:
```toml
[project.scripts]
train = "myproject.scripts:train_command"
mlflow_server = "hydraxcel.logging:run_mlflow_server"
```
