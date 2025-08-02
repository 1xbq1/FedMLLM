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

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score, roc_auc_score, f1_score

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

def run_inference(args):
    model_type=  "openbmb/MiniCPM-V-2_6-int4"
    path_to_adapter=f"./output/output__lora/checkpoint-{args.epoch}"

    model =  AutoModel.from_pretrained(
        model_type,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(
        model,
        path_to_adapter,
        device_map="auto",
        trust_remote_code=True
    ).eval().cuda()

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
    
    f1 = f1_score(truth_list, pred_list, average='macro')*100
    print('F1', f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiple-Choice Video QA Evaluation Script.')

    parser.add_argument('--model-path', default='')
    parser.add_argument('--video-folder', default='../../data/crisis-mmd/raw_data')
    parser.add_argument('--test-csv', default='../../data/crisis-mmd/raw_data/crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=40)
    args = parser.parse_args()

    run_inference(args)
