import glob
import json
import logging
import os
import copy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union, Literal, Tuple
from types import MethodType
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import math
from bitsandbytes import functional as bnb

import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from transformers import AutoModel, AutoTokenizer
from transformers.integrations import deepspeed
from transformers import AutoModel, AutoTokenizer

from dataset import SupervisedDataset, data_collator
from trainer import CPMTrainer, CPMTrainerReg, CPMTrainerSign
from federated_learning import *

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict, set_peft_model_state_dict

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    llm_type: str = field(default="minicpm")
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None


@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedsign", metadata={"help": "the algorithm to use"})
    mu_w: Optional[float] = field(default=0.1, metadata={"help": "the weight of regularization"})
    s_layer: Optional[int] = field(default=4, metadata={"help": "the number of regularization layers"})
    num_rounds: Optional[int] = field(default=1, metadata={"help": "the number of rounds"})
    num_clients: Optional[int] = field(default=10, metadata={"help": "the number of clients"})
    sample_clients: Optional[int] = field(default=10, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="noniid", metadata={"help": "the split strategy"})
    init_learning_rate: Optional[float] = field(default=0.01, metadata={"help": "the initial learning rate"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    lam_sign: Optional[float] = field(default=0.1, metadata={"help": "the weight parameter for fedsign"})
    modality_num: Optional[int] = field(default=2, metadata={"help": "the number of modality or combines"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    save_model_freq: Optional[int] = field(default=5, metadata={"help": "the frequency to save the model. 50 means save every 50 rounds"})


local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir,)

def cosine_warm_learning_rate(last_round, base_lr, total_rounds, warmup_rounds=None, eta_min=0.0):
    if last_round < warmup_rounds:
        current_lr = base_lr * (last_round + 1) / warmup_rounds
    else:
        current_lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * (last_round - warmup_rounds) / (total_rounds - warmup_rounds))) / 2.0
    return current_lr

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path,
    transform,
    data_collator=None,
    llm_type="minicpm",
    slice_config=None,
    patch_size=14,
    query_nums=64,
    batch_vision=False,
    max_length=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    train_json = json.load(open(data_path, "r"))
    train_dataset = dataset_cls(
        train_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=max_length,
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator= partial(data_collator, max_length=max_length),
    )


def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {'Total': all_param, 'Trainable': trainable_params}

def get_low_rank_sign(W, r=8):
    # SVD：W ≈ U @ diag(S) @ V^T
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    U_r = U[:, :r]          # (d_output, r)
    S_r = S[:r]             # (r,)
    Vt_r = Vt[:r, :]        # (r, d_input)

    A = torch.diag(torch.sqrt(S_r)) @ Vt_r  # (r, d_input)
    B = U_r @ torch.diag(torch.sqrt(S_r))   # (d_output, r)

    R_A = torch.sign(A).cuda()     # (r, d_input)
    R_B = torch.sign(B).cuda()     # (d_output, r)

    return R_A, R_B, A.cuda(), B.cuda()

local_rank = 0

def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, FedArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        lora_args,
        fed_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None) :
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    if not training_args.tune_vision:
        model.vpm.requires_grad_(False)
    if not training_args.tune_llm:
        model.llm.requires_grad_(False)

    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")

        rank0_print("Currently using LoRA for fine-tuning the MiniCPM-V model.")
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
        modules_to_save = ['embed_tokens','resampler']
        if training_args.tune_vision:
            modules_to_save.append('vpm')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # generate sign_RA, sign_RB
    model_undeq = AutoModel.from_pretrained(
        "openbmb/MiniCPM-V-2_6",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )
#     for name, param in model_undeq.state_dict().items():
#         print(name, param.shape)

    q_proj = model_undeq.llm.model.layers[27].self_attn.q_proj.weight.detach().to(torch.float32)
    del model_undeq
    torch.cuda.empty_cache()

    _, sign_RB, _, RB = get_low_rank_sign(q_proj, r=8)
    # print("sign_RA.shape", sign_RA.shape)
    # print("RA.shape", RA.shape)
    print("sign_RB.shape", sign_RB.shape)

    # ===== Define the global and local models =====
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))

    for key in global_dict.keys():
        if 'lora_B' in key:
            global_dict[key] = RB
            print("lora_B", key)

    local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

#random initial
#     for key in global_dict.keys():
#         if 'lora_A' in key:
#             sign_RA = torch.sign(torch.randn(global_dict[key].shape[0], global_dict[key].shape[1])).cuda()
#             print('sign_RA', key)
#             print('sign_RA.shape', sign_RA.shape)
#         elif 'lora_B' in key:
#             sign_RB = torch.sign(torch.randn(global_dict[key].shape[0], global_dict[key].shape[1])).cuda()
#             print('sign_RB', key)
#             print('sign_RB.shape', sign_RB.shape)

    rank0_print(get_parameter_number(model))

    llm_type = training_args.llm_type

    rank0_print(f'llm_type={llm_type}')


    # Load data
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    transform_func = build_transform()

    # ===== Split the dataset into clients =====
    local_datasets = []
    sample_num_list = []
    for i in range(fed_args.num_clients):
        data_path = os.path.join(data_args.data_path, f"client_{i}.json")
        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_path=data_path,
            transform=transform_func,
            data_collator=data_collator,
            slice_config=slice_config,
            llm_type=llm_type,
            patch_size=model.config.patch_size,
            query_nums=model.config.query_num,
            batch_vision=batch_vision,
            max_length=training_args.model_max_length,
        )
        local_datasets.append(data_module)
        sample_num_list.append(len(data_module['train_dataset']))

    training_loss = [[] for i in range(fed_args.num_clients)]
    global_save = True
    for round in tqdm(range(fed_args.num_rounds)):
        clients_this_round = get_clients_this_round(fed_args, round)
        print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        for client in range(fed_args.num_clients):
            if client not in clients_this_round:
                training_loss[client].append(-1)            # -1 is an indicator of not training
                continue

            set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

            for name, param in model.named_parameters():
                if 'lora' in name and param.requires_grad:
                    print(f"{name} is trainable")

            sub_dataset = local_datasets[client]
            #new_lr = cosine_warm_learning_rate(round, fed_args.init_learning_rate, fed_args.num_rounds, fed_args.num_rounds*0.01)
            #training_args.learning_rate = new_lr
            training_args.gradient_checkpointing_kwargs={"use_reentrant":False}
            training_args.output_dir = f'../output/output__lora/client-{client}'

            #trainable_params = [p for p in model.parameters() if p.requires_grad]
            #optimizer = AdamW(trainable_params, lr=training_args.learning_rate)

            #training_args.gradient_checkpointing_kwargs={"use_reentrant":False}



            if fed_args.fed_alg == 'fedsign' or 'fedsign' in fed_args.fed_alg:
                trainer = CPMTrainerSign(
                    sign_RA=None,
                    sign_RB=sign_RB,
                    num_train_samples=sample_num_list[client],
                    lam_sign=fed_args.lam_sign,
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **sub_dataset,
                )
            else:
                trainer = CPMTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **sub_dataset,
                )

            results = trainer.train()
            training_loss[client].append(results.training_loss)
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))

            if global_save:
                set_peft_model_state_dict(model, global_dict)

                trainer.save_state()

                safe_save_model_for_hf_trainer(
                    trainer=trainer,
                    output_dir=os.path.join(training_args.output_dir, f"global-lora"),
                    bias=lora_args.lora_bias
                )
                global_save = False

        '''global_dict, global_auxiliary = global_aggregate(
            fed_args, global_dict, local_dict_list, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
        )'''
        # set_peft_model_state_dict(model, global_dict)   # Update global model

        # if (round+1) % fed_args.save_model_freq == 0:
        # if True:
        #     trainer.save_state()

        #     safe_save_model_for_hf_trainer(
        #         trainer=trainer,
        #         output_dir=os.path.join(training_args.output_dir, f"global-lora"),
        #         bias=lora_args.lora_bias)

        np.save(os.path.join(training_args.output_dir, "training_loss.npy"), np.array(training_loss))

if __name__ == "__main__":
    train()
