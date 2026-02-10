import copy
import os
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import requests

from hisim.utils import get_logger


logger = get_logger("hisim")

_all_models_: Dict[str, "ModelInfo"] = {}

model_config_mapping_dict = {
    "num_hidden_layers": ["n_layers", "num_layers", "n_layer"],
    "intermediate_size": [
        "d_ff",
        "n_inner",
        "ffn_hidden_size",
        "dff",
        "dim_ff",
        "ffn_dim",
        "d_inner",
    ],
    "num_attention_heads": ["n_head", "num_heads", "n_heads", "attention_heads"],
    "hidden_size": ["n_embd", "d_model"],
    "max_position_embeddings": ["n_positions"],
    "vocab_size": ["padded_vocab_size"],
    "name": [],
    "model_type": [],
    "torch_dtype": [],
    "hidden_act": ["activation_function"],
    "use_cache": [],
    "use_flash_attn": [],
    "num_key_value_heads": [
        "n_head_kv",
        "num_kv_heads",
        "multi_query_group_num",
    ],  # for qwen 2.5 & llama 3.1
    "kv_hidden_dim": ["kv_channels"],
    "transformers_version": [],
    "max_seq_len": [
        "max_position_embeddings",
        "seq_length",
        "n_positions",
        "position_bias_max_distance",
    ],
    "attn_algo": [],
    "architecture": [],
    "model_path": "",
    ## DeepSeek config
    "moe_intermediate_size": [],
    "n_routed_experts": [
        "num_experts",
        "num_local_experts",
    ],  # Qwen3: num_experts,  gpt-oss: num_local_experts
    "n_shared_experts": [],
    "num_experts_per_tok": [],
    "first_k_dense_replace": [],
    "q_lora_rank": [],
    "kv_lora_rank": [],
    "qk_rope_head_dim": [],
    "qk_nope_head_dim": [],
    "v_head_dim": [],
    "n_group": [],
    "topk_group": [],
    "topk_method": [],
    "ep_size": [],
    "head_dim": [],
    # Qwen/Qwen3-Next-80B-A3B-Instruct
    "attn_linear_conv_kernel_dim": ["linear_conv_kernel_dim"],
    "attn_linear_key_head_dim": ["linear_key_head_dim"],
    "attn_linear_num_key_heads": ["linear_num_key_heads"],
    "attn_linear_num_value_heads": ["linear_num_value_heads"],
    "attn_linear_value_head_dim": ["linear_value_head_dim"],
    # openai/gpt-oss-120b
    "attn_sliding_window": ["sliding_window"],
}


@dataclass
class ModelInfo:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int = 0
    hidden_act: str = ""
    architecture: str = ""
    model_type: str = ""
    max_position_embeddings: int = 4096
    transformers_version: str = ""
    use_cache: bool = True
    attn_algo: str = ""
    torch_dtype: str = "float16"
    use_flash_attn: bool = False
    kv_hidden_dim: int = 0
    num_key_value_heads: int = 0
    max_seq_len: int = 0
    name: str = ""
    model_path: str = ""
    moe_intermediate_size: int = 0  ## intermediate_size of experts
    n_routed_experts: int = 0  ## nums of experts
    n_shared_experts: int = 0  ## shared_experts for all tokens
    num_experts_per_tok: int = 0
    first_k_dense_replace: int = 0  ## if layer id >= first_k_dense_replace, use MOE
    q_lora_rank: int = 0  ## q_lora_rank > 0 means use MLA
    kv_lora_rank: int = 0  ## q_lora_rank > 0 means use MLA
    qk_rope_head_dim: int = 0  ## rope head dim
    qk_nope_head_dim: int = 0  ## nope head dim
    v_head_dim: int = 0  ## v head dim
    ep_size: int = 1
    n_group: int = 0
    topk_group: int = 0
    topk_method: str = ""
    head_dim: int = 0
    quant_method: Optional[str] = None
    # Attention
    num_full_attention: int = 0
    num_linear_attention: int = 0
    attn_linear_conv_kernel_dim: Optional[int] = None
    attn_linear_key_head_dim: Optional[int] = None
    attn_linear_num_key_heads: Optional[int] = None
    attn_linear_num_value_heads: Optional[int] = None
    attn_linear_value_head_dim: Optional[int] = None
    # openai/gpt-oss-120b
    num_sliding_attention: int = 0
    attn_sliding_window: Optional[int] = None

    def __post_init__(self):
        if self.model_type not in [
            "llama",
            "deepseek_v2",
            "deepseek_v3",
            "kimi_k2",
            "gpt_oss",
            "gpt2",
            "qwen",
            "qwen2",
            "qwen3",
            "qwen3_moe",
            "qwen3_next",
            "chatglm",
        ]:
            logger.warning(
                f"Model type '{self.model_type}' is not supported and may cause errors when parsing the configuration."
            )

        self.num_params_embedding = self.hidden_size * self.vocab_size
        self.max_position_embeddings = (
            4096
            if self.max_position_embeddings is None
            else self.max_position_embeddings
        )
        if self.max_seq_len is None or self.max_seq_len == 0:
            self.max_seq_len = self.max_position_embeddings
        if self.intermediate_size is None or self.intermediate_size == 0:
            self.intermediate_size = self.hidden_size * 4
        if self.num_key_value_heads is None or self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None or self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.kv_hidden_dim is None or self.kv_hidden_dim == 0:
            self.kv_hidden_dim = self.num_key_value_heads * self.head_dim

        if self.n_routed_experts != 0 and self.moe_intermediate_size == 0:
            self.moe_intermediate_size = self.intermediate_size

        assert self.num_hidden_layers == (
            self.num_full_attention
            + self.num_linear_attention
            + self.num_sliding_attention
        )

    @staticmethod
    def find_by_model_name(model_name: str) -> Union[None, "ModelInfo"]:
        return _all_models_.get(model_name.upper(), None)

    @staticmethod
    def find_by_prefix(prefix: str) -> Union[None, "ModelInfo"]:
        for k, v in _all_models_.items():
            if k.startswith(prefix.upper()):
                return v
        return None

    @staticmethod
    def list_all_models() -> Dict[str, "ModelInfo"]:
        return _all_models_

    @classmethod
    def from_json(cls, model_path: str) -> Union[None, "ModelInfo"]:
        try:
            json_str = Path(model_path)
            config = json.loads(json_str.read_text())
            config["model_path"] = str(json_str.resolve())
            return cls.from_dict(config)
        except Exception as e:
            logger.error(f"Failed to load model configuration from {model_path}: {e}")
            return None

    @classmethod
    def from_config(cls, config: Dict):
        model_info = cls.find_by_model_name(config.get("name", ""))
        if model_info is not None:
            # deepcopy -> prevent model information from being modified.
            return copy.deepcopy(model_info)
        config["name"] = (
            f"{config.get('name', '')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: Dict, save_to_registry=False):
        model_config = {}
        for k, v in model_config_mapping_dict.items():
            config_v = config.get(k, None)
            if config_v is None:
                for i in v:
                    config_v = config_v if config_v is not None else config.get(i, None)
            if config_v is not None:
                model_config[k] = config_v
        model_config["quant_method"] = config.get("quantization_config", {}).get(
            "quant_method"
        )

        # Adapt the attention layer.
        if (
            config.get("model_type") == "gpt_oss"
            and len(config.get("layer_types", [])) != 0
        ):
            model_config["num_full_attention"] = config.get("layer_types", []).count(
                "full_attention"
            )
            model_config["num_sliding_attention"] = config.get("layer_types", []).count(
                "sliding_attention"
            )
        elif config.get("model_type") == "qwen3_next":
            model_config["num_full_attention"] = config.get(
                "num_hidden_layers"
            ) // config.get("full_attention_interval")
            model_config["num_linear_attention"] = (
                config.get("num_hidden_layers") - model_config["num_full_attention"]
            )
            if config.get("shared_expert_intermediate_size"):
                model_config["n_shared_experts"] = 1
        else:
            model_config["num_full_attention"] = model_config.get("num_hidden_layers")

        # check necessary parameters
        for k in ["hidden_size", "num_attention_heads", "vocab_size", "model_type"]:
            if k not in model_config or model_config[k] is None:
                logger.warning(
                    f"Model configuration type {model_config.get(k, None)} for {k} is not supported."
                )
                return None

        model = cls(**model_config)
        if save_to_registry:
            _all_models_[model.name.upper()] = model
        return model

    @classmethod
    def from_repository(
        cls, url: str, name: str, save_to_registry: bool = False, timeout: float = 3
    ) -> "ModelInfo":
        try:
            config = requests.get(url=url, timeout=timeout).json()
            config["name"] = name
            return cls.from_dict(config, save_to_registry=save_to_registry)
        except Exception as e:
            logger.error(
                f"Failed to retrieve model configuration from {url}. Error: {e}"
            )
            return None

    @classmethod
    def from_huggingface_id(
        cls, model_id: str, save_to_registry: bool = False, timeout: float = 3
    ) -> "ModelInfo":
        if len(model_id.split("/")) != 2:
            logger.warning("Models should be formatted as [organization/name].")
        HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co/").strip("/")
        url = f"{HF_ENDPOINT}/{model_id}/resolve/main/config.json"
        return cls.from_repository(
            url, name=model_id, save_to_registry=save_to_registry, timeout=timeout
        )

    @classmethod
    def from_modelscope_id(
        cls, model_id: str, save_to_registry: bool = False, timeout: float = 3
    ) -> "ModelInfo":
        if len(model_id.split("/")) != 2:
            logger.warning("The model should be formatted as [organization/name]")
        url = f"https://modelscope.cn/models/{model_id}/resolve/master/config.json"
        return cls.from_repository(
            url, name=model_id, save_to_registry=save_to_registry, timeout=timeout
        )
