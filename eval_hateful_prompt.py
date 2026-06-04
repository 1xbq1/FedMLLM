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

# Must match data_gen/gen_data_hateful_missing_prompt.py
PROMPTS = {
    'P1': {
        'preamble': 'Select the best answer to the following multiple-choice question based on the text and image.',
        'question': 'Is the content hateful based on the text and image?',
    },
    'P2': {
        'preamble': 'Select the best answer to the following multiple-choice question based on the available content from the text and image.',
        'question': 'Is the content hateful based on the available content from the text and image?',
    },
    'P3': {
        'preamble': 'Select the best answer to the following multiple-choice question, without considering the modality.',
        'question': 'Is the content hateful, without considering the modality?',
    },
    'P4': {
        'preamble': 'Select the best answer to the following multiple-choice question based on all available information, regardless of its modality.',
        'question': 'Is the content hateful based on all available information, regardless of its modality?',
    },
    'P5': {
        'preamble': 'Select the best answer to the following multiple-choice question.',
        'question': 'Is the content hateful?',
    },
}


def meld_dump(instruct, outputs):
    for idx, output in enumerate(outputs):
        letters = ['A', 'B']
        output = output.replace('answer', '').replace('Answer', '')
        print("output", output)
        pred_answer = re.findall(r'[\(\ ]*[A-G][\)\ ]*', output)
        try:
            assert len(pred_answer) >= 1, \
                'The image instruct: \n"{}"\n output: \n"{}"\n is not in the expected format'.format(instruct, output)
            pred_answer = pred_answer[0].strip().strip('()')
            pred_idx = letters.index(pred_answer)
        except:
            traceback.print_exc()
            pred_idx = 2
    print("pred_id", pred_idx)
    return pred_idx


def run_inference(args):
    assert args.prompt_id in PROMPTS, f"Unknown --prompt-id {args.prompt_id}"
    preamble = PROMPTS[args.prompt_id]['preamble']
    question = PROMPTS[args.prompt_id]['question']
    print(f"[Prompt {args.prompt_id}] preamble: {preamble}")
    print(f"[Prompt {args.prompt_id}] question: {question}")

    model_type = "openbmb/MiniCPM-V-2_6-int4"
    path_to_adapter = f"./output/output__lora/checkpoint-{args.epoch}"

    model = AutoModel.from_pretrained(model_type, trust_remote_code=True)
    model = PeftModel.from_pretrained(
        model, path_to_adapter, device_map="auto", trust_remote_code=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)

    pred_list = []
    truth_list = []
    num_axs = [0 for _ in range(2)]
    truth_axs = [0 for _ in range(2)]

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
        instruct = (
            f"{preamble}\n{text}\n{question}\n"
            f"Options:\n(A) {a0}\n(B) {a1}\n"
            f"Answer with the option's letter from the given choices directly and only give the best option. The best answer is: "
        )
        msgs = [{'role': 'user', 'content': [image, instruct]}]

        try:
            pred = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        except:
            traceback.print_exc()
            pred = 'C'

        pred_id = meld_dump(instruct, [pred])
        pred_list.append(pred_id)
        truth_id = int(label)
        if pred_id == truth_id:
            num_axs[pred_id] += 1
        truth_axs[truth_id] += 1
        print('truth_id', truth_id)
        print('num_axs', num_axs)
        print('truth_axs', truth_axs)
        truth_list.append(truth_id)

    auc = roc_auc_score(truth_list, pred_list) * 100
    print(f'[Prompt {args.prompt_id}] AUC: {auc}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hateful-Memes prompt-variant evaluation.')
    parser.add_argument('--prompt-id', default='P3', choices=list(PROMPTS.keys()),
                        help='Which prompt variant to test (P1..P5).')
    parser.add_argument('--model-path', default='')
    parser.add_argument('--video-folder', default='../../data/hateful_memes/raw_data')
    parser.add_argument('--test-csv', default='../../data/hateful_memes/raw_data/test_seen.jsonl')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=25)
    args = parser.parse_args()

    run_inference(args)
