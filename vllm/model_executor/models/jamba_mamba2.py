# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications by AI21 Labs for Jamba with Mamba2 layers
"""Inference-only Jamba model with Mamba2 (SSD) layers.

This model combines Jamba's hybrid architecture (Mamba+Attention+MoE)
with Mamba2's optimized SSD implementation.
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaMLP as JambaMLP
from vllm.sequence import IntermediateTensors

from .interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class JambaMamba2Config(PretrainedConfig):
    """Configuration class for Jamba with Mamba2 layers.
    
    Extends standard Jamba config with Mamba2-specific parameters.
    """
    model_type = "jamba_mamba2"
    
    # This allows loading without errors for unknown config keys
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class JambaMoE(nn.Module):
    """Mixture of Experts layer for Jamba."""

    def __init__(
        self,
        config: PretrainedConfig,
        num_experts: int | None = None,
        top_k: int | None = None,
        params_dtype: torch.dtype | None = None,
        tp_size: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_total_experts = num_experts or config.num_experts
        self.top_k = top_k or config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if self.num_total_experts > 1:
            self.router = ReplicatedLinear(
                self.hidden_size,
                self.num_total_experts,
                bias=False,
                quant_config=None,
                params_dtype=params_dtype,
            )

        self.experts = FusedMoE(
            self.num_total_experts,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            tp_size=tp_size,
            params_dtype=params_dtype,
            reduce_results=True,
            renormalize=False,
            use_grouped_topk=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (batch * sequence_length, n_experts)
        if self.num_total_experts > 1:
            router_logits, _ = self.router(hidden_states)
        else:
            router_logits = torch.ones(
                (hidden_states.shape[0], 1),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        hidden_states = self.experts(hidden_states, router_logits)
        return hidden_states.view(orig_shape)


class MambaWrapper(nn.Module):
    """Wrapper to create mamba.mixer path structure for checkpoint compatibility."""
    
    def __init__(self, mixer: MambaMixer2):
        super().__init__()
        self.mixer = mixer


class JambaMamba2DecoderLayer(nn.Module):
    """Jamba decoder layer with Mamba2 (SSD) mixer."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        is_lora_enabled: bool | None = False,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Get Mamba2-specific config values
        # These can come from conversion metadata or config
        mamba2_nheads = getattr(config, 'mamba2_nheads', 
                                getattr(config, 'mamba_n_heads', 80))
        mamba2_headdim = getattr(config, 'mamba2_headdim',
                                 getattr(config, 'mamba_d_head', 64))
        mamba2_ngroups = getattr(config, 'mamba2_ngroups',
                                 getattr(config, 'mamba_n_groups', 1))
        mamba2_d_state = getattr(config, 'mamba2_d_state',
                                 getattr(config, 'mamba_d_state', 64))
        
        # Use MambaMixer2 (Mamba2/SSD implementation) wrapped in MambaWrapper
        # to create mamba.mixer path for checkpoint compatibility
        mixer = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=mamba2_d_state,
            conv_kernel_size=config.mamba_d_conv,
            intermediate_size=config.mamba_expand * config.hidden_size,
            use_conv_bias=getattr(config, 'mamba_conv_bias', True),
            use_bias=getattr(config, 'mamba_proj_bias', False),
            n_groups=mamba2_ngroups,
            num_heads=mamba2_nheads,
            head_dim=mamba2_headdim,
            rms_norm_eps=config.rms_norm_eps,
            activation=config.hidden_act,
            use_rms_norm=True,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mamba.mixer",
        )
        self.mamba = MambaWrapper(mixer)

        # MLP/MoE layer
        num_experts = getattr(config, 'layers_num_experts', [1] * config.num_hidden_layers)[layer_idx]
        if num_experts > 1:
            self.feed_forward = JambaMoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )
        else:
            self.feed_forward = JambaMLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Mamba2 forward - vLLM MambaMixer2 handles caching internally
        output = self.mamba.mixer(hidden_states)
        
        # Feed-forward
        hidden_states, residual = self.pre_ff_layernorm(output, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class JambaMamba2AttentionDecoderLayer(nn.Module):
    """Jamba attention decoder layer (unchanged from original Jamba)."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

        # MLP/MoE layer
        num_experts = getattr(config, 'layers_num_experts', [1] * config.num_hidden_layers)[layer_idx]
        if num_experts > 1:
            self.feed_forward = JambaMoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )
        else:
            self.feed_forward = JambaMLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.feed_forward",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
        )
        # Feed-forward
        hidden_states, residual = self.pre_ff_layernorm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": JambaMamba2AttentionDecoderLayer,
    "mamba": JambaMamba2DecoderLayer,
}


@support_torch_compile
class JambaMamba2Model(nn.Module):
    """Jamba model backbone with Mamba2 layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        extra_kwargs = {"is_lora_enabled": bool(vllm_config.lora_config)}

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[layer_idx]]
            return layer_class(
                config,
                layer_idx,
                model_config,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
                **extra_kwargs,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=getattr(self.config, 'num_experts', 1),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            
            # Handle A_log -> A conversion for Mamba2
            if "A_log" in name:
                name = name.replace("A_log", "A")
                
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class JambaMamba2ForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
):
    """Jamba model with Mamba2 layers for causal language modeling.
    
    This model combines:
    - Jamba's hybrid architecture (Mamba+Attention+MoE)
    - Mamba2's SSD (Structured State Space Duality) implementation
    
    Load with: vLLM engine pointing to a converted checkpoint with
    architectures=["JambaMamba2ForCausalLM"]
    """
    
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".self_attn.": ".", ".A_log": ".A"},
    )
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj": ["in_proj"],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        
        assert not cache_config.enable_prefix_caching, \
            "JambaMamba2 currently does not support prefix caching"

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = scheduler_config
        
        self.model = JambaMamba2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class JambaMamba2ForSequenceClassification(JambaMamba2ForCausalLM):
    """Jamba with Mamba2 for sequence classification."""
    
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        num_labels: int = config.num_labels
        score_bias: bool = getattr(config, "score_bias", False)

        self.score = nn.Linear(
            config.hidden_size,
            num_labels,
            bias=score_bias,
            dtype=vllm_config.model_config.head_dtype,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler(
            {
                "token_classify": Pooler.for_token_classify(
                    pooler_config, classifier=self.score
                ),
                "classify": Pooler.for_classify(
                    pooler_config, classifier=self.score, act_fn="classify"
                ),
                "score": Pooler.for_classify(
                    pooler_config, classifier=self.score, act_fn="score"
                ),
            }
        )

