import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import os
import re
import json
import argparse
import traceback

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score, roc_auc_score

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


def run_inference(args):
    model_type=  "openbmb/MiniCPM-V-2_6-int4"
    path_to_adapter="./output/output__lora/checkpoint-50"

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
        aug = ', without considering the modality'
        question = f'Is the content hateful{aug}?'
        instruct = f'Select the best answer to the following multiple-choice question{aug}.\n{text}\n{question}\nOptions:\n(A) {a0}\n(B) {a1}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: '
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
    parser = argparse.ArgumentParser(description='Multiple-Choice Video QA Evaluation Script.')

    parser.add_argument('--model-path', default='')
    parser.add_argument('--video-folder', default='../../data/hateful_memes/raw_data')
    parser.add_argument('--test-csv', default='../../data/hateful_memes/raw_data/test_seen.jsonl')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)
