import dataclasses

import torch
import torch.optim as optim
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

# from optimum.bettertransformer import BetterTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    MT5ForConditionalGeneration,
)
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from configs import fsdp_config as FSDP_CONFIG
from configs.configs_utils import generate_peft_config, update_config
from models.distillation_model import DistillationModel
from models.fsdp import fsdp_auto_wrap_policy
from models.tools import freeze_transformer_layers, get_policies, print_model_size
from policies import AnyPrecisionAdamW, apply_fsdp_checkpointing


def load_tokenizer(name, encoder_decoder):
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = (
            "<|endoftext|>"  # 这里可以根据您的模型需要设置适当的结束符
        )
    if not encoder_decoder:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(train_config, rank):
    use_cache = False if train_config.enable_fsdp else True

    # Define the 4-bit BitsAndBytes config
    q_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    def load():
        if "mt0" in train_config.model_name:
            load_kwargs = dict(use_cache=use_cache)
            if train_config.quantization:
                load_kwargs["quantization_config"] = q_config  # Applied q_config
                load_kwargs["device_map"] = "auto"
            return MT5ForConditionalGeneration.from_pretrained(
                train_config.model_name, **load_kwargs
            )

        elif "Qwen" in train_config.model_name:
            load_kwargs = dict(
                use_cache=use_cache,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
                if getattr(train_config, "pure_bf16", False) else torch.float32,
            )
            if train_config.quantization:
                load_kwargs["quantization_config"] = q_config  # Applied q_config
                load_kwargs["device_map"] = "auto"
            return AutoModelForCausalLM.from_pretrained(
                train_config.model_name, **load_kwargs
            )

        elif "gemma" in train_config.model_name.lower():
            load_kwargs = dict(
                use_cache=use_cache,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
                if getattr(train_config, "pure_bf16", False) else torch.float32,
            )
            if train_config.quantization:
                load_kwargs["quantization_config"] = q_config  # Applied q_config
                load_kwargs["device_map"] = "auto"
            return AutoModelForCausalLM.from_pretrained(
                train_config.model_name, **load_kwargs
            )

        else:
            load_kwargs = dict(
                use_cache=use_cache,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
                if getattr(train_config, "pure_bf16", False)
                else torch.float32,
            )
            if train_config.quantization:
                load_kwargs["quantization_config"] = q_config  # Applied q_config
                load_kwargs["device_map"] = "auto"
            return AutoModelForCausalLM.from_pretrained(
                train_config.model_name, **load_kwargs
            )

    if not train_config.enable_fsdp:
        model = load()

    elif train_config.enable_fsdp:
        if train_config.low_cpu_fsdp:
            if rank == 0:
                model = load()
            else:
                cfg = AutoModelForCausalLM.from_pretrained(
                    train_config.model_name, trust_remote_code=True
                ).config
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(cfg)
                model.use_cache = use_cache
        else:
            model = load()

        if train_config.use_fast_kernels:
            pass

    print_model_size(model, train_config, rank)
    return model


def set_model(model, train_config, fsdp_config, rank, kwargs):
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.use_peft:
        # Applied exact l_config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif train_config.freeze_layers:
        freeze_transformer_layers(train_config.num_freeze_layers)

    if train_config.enable_fsdp:
        if fsdp_config.pure_bf16:
            model.to(torch.bfloat16)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(
            model,
            [LlamaDecoderLayer, GPTNeoXLayer, MistralDecoderLayer, FalconDecoderLayer],
        )

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy
            if train_config.use_peft
            else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True)
            if fsdp_config.fsdp_cpu_offload
            else None,
            mixed_precision=mixed_precision_policy
            if not fsdp_config.pure_bf16
            else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: (
                module.to_empty(device=torch.device("cuda"), recurse=False)
                if train_config.low_cpu_fsdp and rank != 0
                else None
            ),
        )

        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
        return model
    else:
        if train_config.quantization:
            return model
        else:
            return model.to(f"cuda:{rank}")


def get_model(train_config, fsdp_config, rank, kwargs):
    model = load_model(train_config, rank)
    model = set_model(model, train_config, fsdp_config, rank, kwargs)
    tokenizer = load_tokenizer(train_config.model_name, train_config.encoder_decoder)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def get_distillation_models(train_config, distil_config, fsdp_config, rank, kwargs):
    student_tokenizer, student_model = get_model(
        train_config, fsdp_config, rank, kwargs
    )

    teacher_fsdp_config = FSDP_CONFIG()
    update_config((teacher_fsdp_config), **dataclasses.asdict(distil_config))
    teacher_tokenizer, teacher_model = get_model(
        distil_config, distil_config, rank, kwargs
    )

    return (
        student_tokenizer,
        teacher_tokenizer,
        DistillationModel(
            student_model, teacher_model, teacher_tokenizer, student_tokenizer
        ),
    )


def get_optimizer(model, train_config, fsdp_config):
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        return AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        return optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
