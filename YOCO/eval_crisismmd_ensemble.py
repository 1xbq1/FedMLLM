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

import sys
sys.path.append('./')

torch.manual_seed(0)

def meld_dump(instruct, outputs):
    for idx, output in enumerate(outputs):
        instruct = instruct
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        output = output.replace('answer', '')
        output = output.replace('Answer', '')
        print("output", output)
        pred_answer = re.findall('[\(\ ]*[A-H][\)\ ]*', output)
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

def remove_url(text):
    text = re.sub(r'http\S+', '', text)
    return(text)

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
    video_folder: Optional[str] = field(default="../../data/crisis-mmd/raw_data", metadata={"help": "video folder"})
    test_csv: Optional[str] = field(default="../../data/crisis-mmd/raw_data/crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv", metadata={"help": "test csv"})
    output: Optional[str] = field(default="output", metadata={"help": "output folder"})
    epoch: Optional[int] = field(default=0, metadata={"help": "epoch"})

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

#     model_type=  "openbmb/MiniCPM-V-2_6-int4"
#     folder_path = f"./output/output__lora/{args.model_path}"
#     sorted_checkpoints = get_sorted_checkpoints(folder_path)
#     path_to_adapter=f"./output/output__lora/{args.model_path}/{sorted_checkpoints[args.epoch]}"
#     print("loading", path_to_adapter)
    # path_to_adapter="./output/output__lora/checkpoint-1"

    model_type=  "openbmb/MiniCPM-V-2_6-int4"
    model_base =  AutoModel.from_pretrained(
        model_type,
        trust_remote_code=True
    )

#     modules_to_save = ['embed_tokens','resampler']
#     lora_config = LoraConfig(
#         r=lora_args.lora_r,
#         lora_alpha=lora_args.lora_alpha,
#         target_modules=lora_args.lora_target_modules,
#         lora_dropout=lora_args.lora_dropout,
#         bias=lora_args.lora_bias,
#         layers_to_transform=lora_args.lora_layers_to_transform,
#         modules_to_save=modules_to_save,
#     )
#     model = get_peft_model(model_base, lora_config)
#     global_dict = copy.deepcopy(get_peft_model_state_dict(model))
#     # local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
#     proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
#     global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

#     model = PeftModel.from_pretrained(
#         model,
#         path_to_adapter,
#         device_map="auto",
#         trust_remote_code=True
#     ).eval().cuda()

    local_dict_list = []
    for i in range(10):
        folder_path = f"./output/output__lora/client-{i}"
        sorted_checkpoints = get_sorted_checkpoints(folder_path)
        path_to_adapter=f"./output/output__lora/client-{i}/{sorted_checkpoints[args.epoch]}"
        #path_to_adapter=f"./output/output__lora/checkpoint-client-{i}"
        model = PeftModel.from_pretrained(
            model_base,
            path_to_adapter,
            device_map="auto",
            trust_remote_code=True
        ).eval().cuda()
        local_dict_list.append(copy.deepcopy(get_peft_model_state_dict(model)))

#     clients_this_round = [i for i in range(10)]
#     sample_num_list = [1180, 2219, 195, 407, 1873, 229, 4890, 474, 1028, 1113]
#     global_dict, global_auxiliary = global_aggregate(
#         fed_args, global_dict, local_dict_list, sample_num_list, \
#         clients_this_round, 0, proxy_dict=proxy_dict, \
#         opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
#     )
#     set_peft_model_state_dict(model, global_dict)

    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)

    pred_list = []
    truth_list = []
    num_axs = [0 for _ in range(8)]
    truth_axs = [0 for _ in range(8)]

    test_csv_data = pd.read_csv(args.test_csv, sep='\t')
    ax = ['affected_individuals',
          'infrastructure_and_utility_damage',
          'injured_or_dead_people',
          'missing_or_found_people',
          'rescue_volunteering_or_donation_effort',
          'vehicle_damage',
          'other_relevant_information',
          'not_humanitarian']
    options = ''
    for oidx, oax_idx in enumerate(ax):
        options += '\n(' + chr(ord('A') + oidx) + ') ' + oax_idx
    for i in tqdm(np.arange(test_csv_data.shape[0])):
        label_text = test_csv_data['label_image'].iloc[i]
        for idx, ax_idx in enumerate(ax):
            if ax_idx == label_text:
                label = idx
                break
        text = remove_url(test_csv_data['tweet_text'].iloc[i]).strip()
        image_path = '../../data/crisis-mmd/raw_data/'+test_csv_data['image'].iloc[i]
        image = Image.open(image_path).convert('RGB')

        question = 'What is the humanitarian category based on the image and text?'
        instruct = f"Select the best answer to the following multiple-choice question based on the text and image.\n{text}\n{question}\nOptions:{options}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
        msgs = [{'role': 'user', 'content': [image, instruct]}]

        all_preds = [0,0,0,0,0,0,0,0]
        for client_id in range(10):
            set_peft_model_state_dict(model, local_dict_list[client_id])
            try:
                pred = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer
                )
            except:
                traceback.print_exc()
                pred = 'C'
            pred_id_m = meld_dump(instruct, [pred])
            all_preds[pred_id_m] += 1
        pred_id = all_preds.index(max(all_preds))

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
    
    f1 = f1_score(truth_list, pred_list, average='macro')*100
    print('F1', f1)

if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Multiple-Choice Video QA Evaluation Script.')
#
#     parser.add_argument('--model-path', default='')
#     parser.add_argument('--video-folder', default='../../data/crisis-mmd/raw_data')
#     parser.add_argument('--test-csv', default='../../data/crisis-mmd/raw_data/crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv')
#     parser.add_argument("--batch-size", type=int, default=1)
#     parser.add_argument("--num-workers", type=int, default=8)
#     parser.add_argument("--epoch", type=int, default=0)
#     args = parser.parse_args()

    run_inference()

