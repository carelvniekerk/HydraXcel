# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HydraXcel
# Author: Carel van Niekerk
# Year: 2026
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT Codex, Claude Code, Gemini.
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
"""Accelerate Configuration Registry."""

from dataclasses import dataclass, field

from hydraxcel.hydra import config_store, hydra_config

__all__ = ["load_accelerate_configs"]


@dataclass
class HardwareConfig:
    """Hardware Configuration."""

    cpu: bool = False
    multi_gpu: bool = False
    tpu: bool = False
    num_processes: int = 1
    num_machines: int = 1
    num_cpu_threads_per_process: int = 2
    enable_cpu_affinity: bool = False


@dataclass
class CompileConfig:
    """Compile Configuration."""

    dynamo_backend: str = "no"
    dynamo_mode: str = "default"
    dynamo_use_fullgraph: bool = False
    dynamo_use_dynamic: bool = False
    dynamo_use_regional_compilation: bool = False


@dataclass
class ParadigmConfig:
    """Paradigm Configuration."""

    use_deepspeed: bool = False
    use_fsdp: bool = False
    use_parallelism_config: bool = False
    use_megatron_lm: bool = False


@dataclass
class DeepSpeedConfig(ParadigmConfig):
    """DeepSpeed Configuration."""

    use_deepspeed: bool = True

    deepspeed_config_file: str | None = (
        None  # Provide a JSON to override below auto flags
    )
    zero_stage: int = 2  # ZeRO Stage 2: good memory savings + low complexity
    offload_optimizer_device: str | None = None  # 'cpu' only if still OOM after Stage 2
    offload_param_device: str | None = None  # 'cpu' only if still OOM (adds latency)
    offload_optimizer_nvme_path: str | None = None  # NVMe path for extreme memory
    offload_param_nvme_path: str | None = None
    gradient_accumulation_steps: int = 1  # Raise (e.g. 4/8) for larger effective batch
    gradient_clipping: float = 1.0  # Stabilize large model training
    zero3_init_flag: bool | None = None  # Set 'true' when moving to ZeRO-3
    zero3_save_16bit_model: bool | None = None
    deepspeed_hostfile: str | None = None  # Populate for explicit multi-node host
    deepspeed_exclusion_filter: str | None = None
    deepspeed_inclusion_filter: str | None = None
    deepspeed_multinode_launcher: str | None = None
    deepspeed_moe_layer_cls_names: str | None = None


@dataclass
class TorchFSDPConfig(ParadigmConfig):
    """Torch FSDP Configuration."""

    use_fsdp: bool = True

    fsdp_version: str = "1"
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_offload_params: str = "false"
    fsdp_min_num_params: float = 1e8  # Auto-wrap threshold (adjust for model size)
    fsdp_reshard_after_forward: str = "true"  # Free full params ASAP to save memory
    fsdp_forward_prefetch: str = "false"  # Enable later if profiling shows benefit
    fsdp_use_orig_params: str = "true"  # Keeps original param objects (safer debugging)
    fsdp_cpu_ram_efficient_loading: str = "true"  # Stream load large checkpoints
    fsdp_sync_module_states: str = "true"  # Sync on init (multi-rank consistent)
    fsdp_activation_checkpointing: str = "false"
    fsdp_auto_wrap_policy: str | None = None
    fsdp_transformer_layer_cls_to_wrap: str | None = None
    fsdp_backward_prefetch: str | None = None
    fsdp_state_dict_type: str | None = None


@dataclass
class MixedPrecisionConfig:
    """Mixed Precision Configuration."""

    mixed_precision: str = "no"


@dataclass
class FP8PrecisionConfig:
    """FP8 Precision Configuration."""

    mixed_precision: str = "fp8"
    fp8_backend: str = "NO"
    fp8_format: str = "HYBRID"
    fp8_amax_compute_algo: str = "most_recent"
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1024
    fp8_use_autocast_during_eval: bool = False
    fp8_override_linear_precision: list[bool] = field(
        default_factory=lambda: [False, False, False],
    )
    fp8_opt_level: str = "O2"


AccelerateConfig = hydra_config(
    "AccelerateConfig",
    defaults=[
        {"accelerate/hardware": "gpu"},
        {"accelerate/compile": "torch"},
        {"accelerate/paradigm": "torch-ddp"},
        {"accelerate/mixed-precision": "bf16"},
    ],
    add_training_script_placeholder=True,
)


def load_accelerate_configs() -> None:
    """Load accelerate configurations into the config store."""
    config_store.store(
        name="accelerate",
        node=AccelerateConfig,
    )
    config_store.store(
        name="cpu",
        group="accelerate/hardware",
        node=HardwareConfig(cpu=True),
    )
    config_store.store(
        name="gpu",
        group="accelerate/hardware",
        node=HardwareConfig(),
    )
    config_store.store(
        name="multi-gpu",
        group="accelerate/hardware",
        node=HardwareConfig(multi_gpu=True, num_processes=2),
    )
    config_store.store(
        name="none",
        group="accelerate/compile",
        node=CompileConfig(),
    )
    config_store.store(
        name="torch",
        group="accelerate/compile",
        node=CompileConfig(
            dynamo_backend="inductor",
            dynamo_mode="default",
            dynamo_use_dynamic=True,
        ),
    )
    config_store.store(
        name="inductor",
        group="accelerate/compile",
        node=CompileConfig(dynamo_backend="inductor"),
    )
    config_store.store(
        name="torch-ddp",
        group="accelerate/paradigm",
        node=ParadigmConfig(),
    )
    config_store.store(
        name="deepspeed",
        group="accelerate/paradigm",
        node=DeepSpeedConfig(),
    )
    config_store.store(
        name="deepspeed-zero3",
        group="accelerate/paradigm",
        node=DeepSpeedConfig(
            zero_stage=3,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_init_flag=True,
            zero3_save_16bit_model=True,
        ),
    )
    config_store.store(
        name="torch-fsdp",
        group="accelerate/paradigm",
        node=TorchFSDPConfig(),
    )
    config_store.store(
        name="no",
        group="accelerate/mixed-precision",
        node=MixedPrecisionConfig(),
    )
    config_store.store(
        name="fp16",
        group="accelerate/mixed-precision",
        node=MixedPrecisionConfig(mixed_precision="fp16"),
    )
    config_store.store(
        name="bf16",
        group="accelerate/mixed-precision",
        node=MixedPrecisionConfig(mixed_precision="bf16"),
    )
    config_store.store(
        name="fp8",
        group="accelerate/mixed-precision",
        node=FP8PrecisionConfig(),
    )
