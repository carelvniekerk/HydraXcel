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
"""Configuration for Accelerate Launcher using Hydra."""

from dataclasses import dataclass, field
from typing import ClassVar

__all__ = ["LaunchConfig"]


@dataclass(init=True, frozen=False)
class LaunchConfig:
    """Launch configuration for Accelerate.

    NOTE: This model is frozen (immutable). To adjust values create a new
    instance. Only documented fields are allowed (extra is forbidden).
    """

    # ---------------------- Allowed value sets (class-level) ------------------
    _MIXED_PRECISION_CHOICES: ClassVar[set[str]] = {"no", "fp16", "bf16", "fp8"}
    _DYNAMO_BACKEND_CHOICES: ClassVar[set[str]] = {
        "no",
        "inductor",
        "eager",
        "aot_eager",
        "nvfuser",
        "aot_cudagraphs",
    }
    _DYNAMO_MODE_CHOICES: ClassVar[set[str]] = {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
        "debug",
    }
    _FSDP_SHARDING_STRATEGY_CHOICES: ClassVar[set[str]] = {
        "FULL_SHARD",
        "SHARD_GRAD_OP",
        "NO_SHARD",
    }
    _FP8_BACKEND_CHOICES: ClassVar[set[str]] = {"NO", "TE", "MSAMP", "AO"}
    _FP8_FORMAT_CHOICES: ClassVar[set[str]] = {"HYBRID", "E4M3", "E5M2"}
    _FP8_AMAX_ALGO_CHOICES: ClassVar[set[str]] = {
        "most_recent",
        "max",
        "moving_average",
        "ema",
    }
    _PARADIGM_FLAGS: ClassVar[tuple[str, ...]] = (
        "use_deepspeed",
        "use_fsdp",
        "use_parallelism_config",
        "use_megatron_lm",
    )

    # Core / meta
    training_script: str = field(default_factory=str)
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
    fp8_override_linear_precision: tuple[bool, bool, bool] = (
        False,
        False,
        False,
    )  # tuple of bools
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
    # Internal-only (populated later, not user initialized / serialized)
    deepspeed_fields_from_accelerate_config: str | None = field(
        default=None,
        init=False,
    )
    use_cpu: bool = field(default=False, init=False)
    use_xpu: bool = field(default=False, init=False)
    nproc_per_node: int = field(default=1, init=False)
    master_port: str = field(default="0000", init=False)

    def __post_init__(self) -> None:
        """Validate the configuration after initialization."""
        # Validate fields
        self._validate_mixed_precision()
        self._validate_dynamo_backend()
        self._validate_dynamo_mode()
        self._validate_fsdp_version()
        self._validate_fsdp_sharding_strategy()
        self._validate_zero_stage()
        self._validate_positive_parallel_degrees(self.megatron_lm_tp_degree)
        self._validate_positive_parallel_degrees(self.megatron_lm_pp_degree)
        self._validate_fp8_backend()
        self._validate_fp8_format()
        self._validate_fp8_amax_algo()
        self._validate_fp8_margin()
        self._validate_fp8_interval()
        self._validate_fp8_history_len()
        self._validate_fp8_override_tuple()

        # Validate cross-field dependencies
        self._validate_paradigm_exclusivity()
        self._validate_deepspeed_block()
        self._validate_megatron_block()
        self._validate_fp8_block()

    def _validate_mixed_precision(self) -> None:
        if self.mixed_precision not in self._MIXED_PRECISION_CHOICES:
            msg = (
                f"mixed_precision must be one of {self._MIXED_PRECISION_CHOICES}, "
                f"got {self.mixed_precision!r}"
            )
            raise ValueError(msg)

    def _validate_dynamo_backend(self) -> None:
        if self.dynamo_backend not in self._DYNAMO_BACKEND_CHOICES:
            msg = (
                f"dynamo_backend must be in {self._DYNAMO_BACKEND_CHOICES}, "
                f"got {self.dynamo_backend!r}"
            )
            raise ValueError(msg)

    def _validate_dynamo_mode(self) -> None:
        if self.dynamo_mode not in self._DYNAMO_MODE_CHOICES:
            msg = (
                f"dynamo_mode must be in {self._DYNAMO_MODE_CHOICES}, "
                f"got {self.dynamo_mode!r}"
            )
            raise ValueError(msg)

    def _validate_fsdp_version(self) -> None:
        if self.fsdp_version not in {"1", "2"}:
            msg = "fsdp_version must be '1' or '2'"
            raise ValueError(msg)

    def _validate_fsdp_sharding_strategy(self) -> None:
        if self.fsdp_sharding_strategy not in self._FSDP_SHARDING_STRATEGY_CHOICES:
            msg = (
                f"fsdp_sharding_strategy must be in "
                f"{self._FSDP_SHARDING_STRATEGY_CHOICES}, "
                f"got {self.fsdp_sharding_strategy!r}"
            )
            raise ValueError(msg)

    def _validate_zero_stage(self) -> None:
        if not self.use_deepspeed:
            return
        if self.zero_stage not in {1, 2, 3}:
            msg = "zero_stage must be 1, 2 or 3 when set"
            raise ValueError(msg)

    @staticmethod
    def _validate_positive_parallel_degrees(value: int) -> None:
        if value < 1:
            msg = "Megatron parallel degrees must be >= 1"
            raise ValueError(msg)

    def _validate_fp8_backend(self) -> None:
        if (
            self.fp8_backend is not None
            and self.fp8_backend not in self._FP8_BACKEND_CHOICES
        ):
            msg = (
                f"fp8_backend must be one of {self._FP8_BACKEND_CHOICES}, "
                f"got {self.fp8_backend!r}"
            )
            raise ValueError(msg)

    def _validate_fp8_format(self) -> None:
        if self.fp8_format not in self._FP8_FORMAT_CHOICES:
            msg = (
                f"fp8_format must be in {self._FP8_FORMAT_CHOICES}, "
                f"got {self.fp8_format!r}"
            )
            raise ValueError(msg)

    def _validate_fp8_amax_algo(self) -> None:
        if self.fp8_amax_compute_algo not in self._FP8_AMAX_ALGO_CHOICES:
            msg = (
                f"fp8_amax_compute_algo must be in {self._FP8_AMAX_ALGO_CHOICES}, "
                f"got {self.fp8_amax_compute_algo!r}"
            )
            raise ValueError(msg)

    def _validate_fp8_margin(self) -> None:
        if self.fp8_margin < 0:
            msg = "fp8_margin must be >= 0"
            raise ValueError(msg)

    def _validate_fp8_interval(self) -> None:
        if self.fp8_interval < 1:
            msg = "fp8_interval must be >= 1"
            raise ValueError(msg)

    def _validate_fp8_history_len(self) -> None:
        if self.fp8_amax_history_len < 1:
            msg = "fp8_amax_history_len must be >= 1"
            raise ValueError(msg)

    def _validate_fp8_override_tuple(self) -> None:
        if len(self.fp8_override_linear_precision) != 3:  # noqa: PLR2004 # Single check that tuple has 3 elements
            msg = "fp8_override_linear_precision must be a 3-tuple of bools"
            raise ValueError(msg)

    # ---- helper segmented validators to reduce complexity ----
    def _validate_paradigm_exclusivity(self) -> None:
        enabled = [getattr(self, f) for f in self._PARADIGM_FLAGS]
        if sum(bool(x) for x in enabled) > 1:
            msg = "Only one of use_deepspeed, use_fsdp, use_parallelism_config, use_megatron_lm may be True"  # noqa: E501
            raise ValueError(msg)

    def _validate_deepspeed_block(self) -> None:
        if self.zero_stage is not None and not self.use_deepspeed:
            raise ValueError("zero_stage provided but use_deepspeed is False")  # noqa: EM101, TRY003
        offload_fields = (
            self.offload_optimizer_device,
            self.offload_param_device,
            self.offload_optimizer_nvme_path,
            self.offload_param_nvme_path,
        )
        if any(x is not None for x in offload_fields) and not self.use_deepspeed:
            raise ValueError("Offload parameters require use_deepspeed=True")  # noqa: EM101, TRY003

    def _validate_megatron_block(self) -> None:
        if self.use_megatron_lm and self.megatron_lm_pp_degree > 1:
            if self.megatron_lm_num_micro_batches is None:
                raise ValueError(  # noqa: TRY003
                    "megatron_lm_num_micro_batches must be set when pp_degree > 1",  # noqa: EM101
                )
            if self.megatron_lm_num_micro_batches < self.megatron_lm_pp_degree * 2:
                raise ValueError(  # noqa: TRY003
                    "megatron_lm_num_micro_batches should be at least 2 * pp_degree for pipeline efficiency",  # noqa: E501, EM101
                )

    def _validate_fp8_block(self) -> None:
        if self.mixed_precision == "fp8":
            if self.fp8_backend is None:
                raise ValueError("fp8_backend must be set when mixed_precision='fp8'")  # noqa: EM101, TRY003
        elif self.fp8_backend is not None:
            msg = "fp8_backend should be None unless mixed_precision='fp8'"
            raise ValueError(msg)
