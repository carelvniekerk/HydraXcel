# coding=utf-8
# --------------------------------------------------------------------------------
# Project: AgenticRL
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
"""Custom launcher for using accelerate."""

from dataclasses import dataclass, field
from typing import Any

from accelerate.commands.launch import launch_command


@dataclass
class LaunchConfig:
    """Configuration container mirroring `accelerate launch` CLI arguments.

    All hyphenated CLI flags are represented in snake_case. Only populate the
    fields you need; unset (None/False) values will defer to accelerate's
    internal defaults when converted to argparse namespace elsewhere.
    """

    # Core / meta
    training_script: str
    training_script_args: list[str] = field(default_factory=list)
    config_file: str | None = None
    quiet: bool = False

    # Hardware selection
    cpu: bool = False
    multi_gpu: bool = False
    tpu: bool = False

    # Resource selection
    mixed_precision: str = "no"  # choices: no|fp16|bf16|fp8
    num_processes: int = 1
    num_machines: int = 1
    num_cpu_threads_per_process: int = 2
    enable_cpu_affinity: bool = False
    dynamo_backend: str = "no"
    dynamo_mode: str = "default"
    dynamo_use_fullgraph: bool = False
    dynamo_use_dynamic: bool = False
    dynamo_use_regional_compilation: bool = False

    # Paradigm selection
    use_deepspeed: bool = False
    use_fsdp: bool = False
    use_parallelism_config: bool = False
    use_megatron_lm: bool = False
    use_xpu: bool | None = None  # deprecated

    # Distributed GPUs
    gpu_ids: str | None = None
    same_network: bool = False
    machine_rank: int | None = None
    main_process_ip: str | None = None
    main_process_port: int | None = None
    tee: str = "0"
    log_dir: str | None = None
    role: str = "default"
    rdzv_backend: str = "static"
    rdzv_conf: str = ""
    max_restarts: int = 0
    monitor_interval: float = 0.1
    module: bool = False
    no_python: bool = False

    # TPU
    tpu_use_cluster: bool = False
    tpu_use_sudo: bool = False
    vm: list[str] | None = field(default_factory=list)
    env: list[str] | None = field(default_factory=list)
    main_training_function: str | None = None
    downcast_bf16: bool = False

    # DeepSpeed
    deepspeed_config_file: str | None = None
    zero_stage: int | None = None
    offload_optimizer_device: str | None = None
    offload_param_device: str | None = None
    offload_optimizer_nvme_path: str | None = None
    offload_param_nvme_path: str | None = None
    gradient_accumulation_steps: int | None = None
    gradient_clipping: float | None = None
    zero3_init_flag: str | None = None  # 'true'|'false'
    zero3_save_16bit_model: str | None = None  # 'true'|'false'
    deepspeed_hostfile: str | None = None
    deepspeed_exclusion_filter: str | None = None
    deepspeed_inclusion_filter: str | None = None
    deepspeed_multinode_launcher: str | None = None
    deepspeed_moe_layer_cls_names: str | None = None

    # FSDP
    fsdp_version: str = "1"  # choices 1|2
    fsdp_offload_params: str = "false"
    fsdp_min_num_params: float = 1e8
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_reshard_after_forward: str = "true"
    fsdp_auto_wrap_policy: str | None = None
    fsdp_transformer_layer_cls_to_wrap: str | None = None
    fsdp_backward_prefetch: str | None = None
    fsdp_state_dict_type: str | None = None
    fsdp_forward_prefetch: str = "false"
    fsdp_use_orig_params: str = "true"
    fsdp_cpu_ram_efficient_loading: str = "true"
    fsdp_sync_module_states: str = "true"
    fsdp_activation_checkpointing: str = "false"

    # Megatron-LM
    megatron_lm_tp_degree: int = 1
    megatron_lm_pp_degree: int = 1
    megatron_lm_num_micro_batches: int | None = None
    megatron_lm_sequence_parallelism: str | None = None
    megatron_lm_recompute_activations: str | None = None
    megatron_lm_use_distributed_optimizer: str | None = None
    megatron_lm_gradient_clipping: float = 1.0

    # FP8
    fp8_backend: str | None = None  # 'te'|'msamp'
    fp8_use_autocast_during_eval: bool = False
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_format: str = "HYBRID"
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "most_recent"
    fp8_override_linear_precision: Any = (False, False, False)  # tuple of bools
    fp8_opt_level: str = "O2"

    # AWS
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # MPI
    mpirun_hostfile: str | None = None
    mpirun_ccl: int = 1

    # ParallelismConfig
    parallelism_config_dp_replicate_size: int = 1
    parallelism_config_dp_shard_size: int = 1
    parallelism_config_tp_size: int = 1
    parallelism_config_cp_size: int = 1
    parallelism_config_cp_comm_strategy: str = "allgather"

    # Misc
    debug: bool = False

    # Internal / computed-only: filled later by accelerate when needed
    deepspeed_fields_from_accelerate_config: str | None = None


def main() -> None:
    """Demonstrate instantiating a LaunchConfig and invoking launch_command.

    NOTE: This does not yet convert to the expected argparse.Namespace that
    `launch_command` requires; additional glue code would be needed.
    """
    config = LaunchConfig(
        training_script="src/test/scripts/hello.py",
        training_script_args=["--name", "Daphy"],
    )
    launch_command(config)  # type: ignore[arg-type]
