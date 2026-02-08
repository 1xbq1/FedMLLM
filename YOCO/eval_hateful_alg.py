import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import os
import re
import json
import argparse
import traceback
import pandas as pd
import numpy as np
import copy
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict, set_peft_model_state_dict

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score, roc_auc_score, f1_score
from finetune.federated_learning import *
import ot  # Python Optimal Transport

import sys
sys.path.append('./')

torch.manual_seed(0)

def meld_dump(instruct, outputs):
    for idx, output in enumerate(outputs):
        instruct = instruct
        letters = ['A', 'B']

        output = output.replace('answer', '')
        output = output.replace('Answer', '')
        print("output", output)
        pred_answer = re.findall('[\(\ ]*[A-G][\)\ ]*', output)
        try:

            assert len(pred_answer) >= 1, 'The image instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(instruct, output)
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
        except:
            traceback.print_exc()
            pred_idx = 2
    print("pred_id", pred_idx)
    return pred_idx

#lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.27\.self_attn\.q_proj"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None

@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedyogi", metadata={"help": "the algorithm to use"})
    w_type: Optional[str] = field(default="base", metadata={"help": "client-wise, layer-wise, B-wise"})
    mu_w: Optional[float] = field(default=0.1, metadata={"help": "the weight of regularization"})
    lin_layer: Optional[int] = field(default=14, metadata={"help": "the layer where linearization begins"})
    s_layer: Optional[int] = field(default=2, metadata={"help": "the number of regularization layers"})
    num_rounds: Optional[int] = field(default=1, metadata={"help": "the number of rounds"})
    num_clients: Optional[int] = field(default=10, metadata={"help": "the number of clients"})
    sample_clients: Optional[int] = field(default=10, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="noniid", metadata={"help": "the split strategy"})
    init_learning_rate: Optional[float] = field(default=0.01, metadata={"help": "the initial learning rate"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    modality_num: Optional[int] = field(default=2, metadata={"help": "the number of modality or combines"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    save_model_freq: Optional[int] = field(default=1, metadata={"help": "the frequency to save the model. 50 means save every 50 rounds"})

@dataclass
class EvalArguments:
    video_folder: Optional[str] = field(default="../../data/hateful_memes/raw_data", metadata={"help": "video folder"})
    test_csv: Optional[str] = field(default="../../data/hateful_memes/raw_data/test_seen.jsonl", metadata={"help": "test csv"})
    output: Optional[str] = field(default="output-hateful-linear-miss0.5", metadata={"help": "output folder"})
    epoch: Optional[int] = field(default=0, metadata={"help": "epoch"})
    ot_flag: bool = field(default=False, metadata={"help": "Use optimal transport."})

def get_sorted_checkpoints(folder_path):
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    checkpoints = []
    for subdir in subdirs:
        match = re.match(r'checkpoint-(\d+)', subdir)
        if match:
            checkpoints.append((subdir, int(match.group(1))))

    checkpoints.sort(key=lambda x: x[1])

    return [checkpoint[0] for checkpoint in checkpoints]


def run_inference():
    parser = transformers.HfArgumentParser(
        (LoraArguments, FedArguments, EvalArguments)
    )

    (
        lora_args,
        fed_args,
        args,
    ) = parser.parse_args_into_dataclasses()

    # size_in_gb = 15
    # size_in_bytes = size_in_gb * 1024**3
    # size_in_floats = size_in_bytes // 4
    # dummy_tensor = torch.empty(size_in_floats, device='cuda', dtype=torch.float32)

    model_type=  "openbmb/MiniCPM-V-2_6-int4"
#     folder_path = f"./output/output__lora/{args.model_path}"
#     sorted_checkpoints = get_sorted_checkpoints(folder_path)
#     path_to_adapter=f"./output/output__lora/{args.model_path}/{sorted_checkpoints[args.epoch]}"
#     #path_to_adapter=f"./output/output__lora/{args.model_path}/checkpoint-{args.epoch}"
#     print("loading", path_to_adapter)
    # path_to_adapter="./output/output__lora/checkpoint-1"

    model_base =  AutoModel.from_pretrained(
        model_type,
        trust_remote_code=True
    )

    # model = PeftModel.from_pretrained(
    #     model,
    #     path_to_adapter,
    #     device_map="auto",
    #     trust_remote_code=True
    # ).eval().cuda()

    modules_to_save = ['embed_tokens','resampler']
    lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
    model = get_peft_model(model_base, lora_config)
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    # local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

    # model = PeftModel.from_pretrained(
    #     model_base,
    #     path_to_adapter,
    #     device_map="auto",
    #     trust_remote_code=True
    # ).eval().cuda()

    local_dict_list = []
    outdir = 'output'
    for i in range(10):
        if 'conflict' in fed_args.fed_alg:
            if '-aftot' in fed_args.fed_alg:
                folder_path = f"./{outdir}/output__lora/client-{i}"
                sorted_checkpoints = get_sorted_checkpoints(folder_path)
                path_to_adapter=f"./{outdir}/output__lora/client-{i}/{sorted_checkpoints[args.epoch]}"
            elif '-befot' in fed_args.fed_alg:
                folder_path = f"./{outdir}/output__lora/client-{i}-ot"
                sorted_checkpoints = get_sorted_checkpoints(folder_path)
                path_to_adapter=f"./{outdir}/output__lora/client-{i}-ot/{sorted_checkpoints[args.epoch]}"
            else:
                folder_path = f"./{outdir}/output__lora/client-{i}"
                sorted_checkpoints = get_sorted_checkpoints(folder_path)
                path_to_adapter=f"./{outdir}/output__lora/client-{i}/{sorted_checkpoints[args.epoch]}"
        elif '-ot' in fed_args.fed_alg:
            folder_path = f"./{outdir}/output__lora/client-{i}-ot"
            sorted_checkpoints = get_sorted_checkpoints(folder_path)
            path_to_adapter=f"./{outdir}/output__lora/client-{i}-ot/{sorted_checkpoints[args.epoch]}"
        else:
            folder_path = f"./{outdir}/output__lora/client-{i}"
            sorted_checkpoints = get_sorted_checkpoints(folder_path)
            path_to_adapter=f"./{outdir}/output__lora/client-{i}/{sorted_checkpoints[args.epoch]}"
        #path_to_adapter=f"./output/output__lora/checkpoint-client-{i}"
        model_mid = PeftModel.from_pretrained(
            model_base,
            path_to_adapter,
            device_map="auto",
            trust_remote_code=True
        ).eval().cuda()
        local_dict_list.append(copy.deepcopy(get_peft_model_state_dict(model_mid)))
    del model_mid
    torch.cuda.empty_cache()

    clients_this_round = [i for i in range(10)]
    #sample_num_list = [257, 2933, 1378, 577, 166, 166, 168, 1354, 704, 797]
    sample_num_list = [523, 1433, 173, 30, 1014, 3, 4285, 123, 507, 409]
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, 0, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model, global_dict)

    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)

    pred_list = []
    truth_list = []
    num_axs = [0 for _ in range(2)]
    truth_axs = [0 for _ in range(2)]
    # Iterate over each sample in the ground truth file
    filenames = []
    with open(args.test_csv, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            filenames.append(data)
    for line in filenames:
        label = line['label']
        text = line['text']

        image_path = os.path.join(args.video_folder, line['img'])
        image = Image.open(image_path).convert('RGB')

        a0 = 'not-hateful'
        a1 = 'hateful'
        question = 'Is the content hateful based on the text and image?'
        instruct = f'Select the best answer to the following multiple-choice question based on the text and image.\n{text}\n{question}\nOptions:\n(A) {a0}\n(B) {a1}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: '
        msgs = [{'role': 'user', 'content': [image, instruct]}]

        try:
            pred = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer
            )
        except:
            traceback.print_exc()
            pred = 'C'

        pred_id = meld_dump(instruct, [pred])
        print("pred_id", pred_id)
        pred_list.append(pred_id)
        truth_id = int(label)
        if pred_id == truth_id:
            num_axs[pred_id] += 1
        truth_axs[truth_id] += 1
        print('truth_id', truth_id)
        print('num_axs', num_axs)
        print('truth_axs', truth_axs)
        truth_list.append(truth_id)

    auc = roc_auc_score(truth_list, pred_list)*100
    print('AUC', auc)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Multiple-Choice Video QA Evaluation Script.')

    # parser.add_argument('--model-path', default='')
    # parser.add_argument('--video-folder', default='../../data/hateful_memes/raw_data')
    # parser.add_argument('--test-csv', default='../../data/hateful_memes/raw_data/test_seen.jsonl')
    # parser.add_argument("--batch-size", type=int, default=1)
    # parser.add_argument("--num-workers", type=int, default=8)
    # parser.add_argument('-a', '--fed-alg', default='fedyogi', help='fedyogi, fedadam, fedavgm, fedadagrad')
    # parser.add_argument('-o', '--output', default='output-hateful-miss0.3')
    # args = parser.parse_args()

    run_inference()
