import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import os
import re
import json
import argparse
import traceback
import logging

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score, roc_auc_score
import torch.distributed as dist
from vqa_slake import MedVQA
from vqa_eval_slake import MedVQAEval

import sys
sys.path.append('./')
torch.manual_seed(0)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def save_result(result, result_dir, filename, remove_duplicate=""):
    result_file = os.path.join(
        result_dir, "%s_rank%d.json" % (filename, get_rank())
    )
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    json.dump(result, open(result_file, "w"))

    if is_dist_avail_and_initialized():
        dist.barrier()

    if is_main_process():
        logging.warning("rank %d starts merging results." % get_rank())
        # combine results from all processes
        result = []

        for rank in range(get_world_size()):
            result_file = os.path.join(
                result_dir, "%s_rank%d.json" % (filename, rank)
            )
            res = json.load(open(result_file, "r"))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, "w"))
        print("result file saved to %s" % final_result_file)

    return final_result_file

def report_metrics(result_file, split):
    """
    Use official VQA evaluation script to report metrics.
     """
    metrics = {}
    # if split in self.ques_files and split in self.anno_files:
    #     vqa = VQA(self.anno_files[split], self.ques_files[split])
    if split == "test":
        test_list = json.load(open('../../data/medical/raw_data/Slake1.0/test.json'))
        vqa = MedVQA(test_list, test_list)
        vqa_result = vqa.loadRes(
            # resFile=result_file, quesFile=self.ques_files[split]
            resFile=result_file, quesFile=test_list
        )
    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    vqa_scorer = MedVQAEval(vqa, vqa_result, n=2)
    logging.info("Start VQA evaluation.")
    vqa_scorer.evaluate()

    # print accuracies
    overall_acc = vqa_scorer.accuracy["overall"]
    metrics["agg_metrics"] = overall_acc

    logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
    logging.info("Per Answer Type Accuracy is the following:")

    for ans_type in vqa_scorer.accuracy["perAnswerType"]:
        logging.info(
            "%s : %.02f"
            % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
        )
        metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "evaluate_slake.txt"), "a"
    ) as f:
        f.write(json.dumps(metrics) + "\n")
    return metrics

def run_inference(args):
    model_type=  "openbmb/MiniCPM-V-2_6-int4"
    path_to_adapter=f"./output/output__lora/checkpoint-{args.model_path}"

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

    with open("../../data/medical/raw_data/Slake1.0/test.json", "r") as f:
        test_data = json.load(f)

    pred_qa_pairs = []
    gpteval_vqa_list = []
    for data in test_data:
        image_path = os.path.join(args.video_folder, data['img_name'])
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, data['question']]}]

        try:
            answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer
            )
        except:
            traceback.print_exc()
            answer = 'Error'
        
        print('qid', data['qid'], 'answer', answer)
        pred_qa_pairs.append({"qid": data['qid'], "answer": answer})
        gpteval_vqa_list.append({'qid': data['qid'],
                                "question": data['question'],
                                "ground_truth": data['answer'],
                                "generated_answer": answer,
                                "answer_type": data['answer_type'],
                                "question_type": data['content_type']})
    
    json.dump(gpteval_vqa_list, open(os.path.join(args.result_dir, "gpteval_vqa_slake.json"), "w"))
    result_file = save_result(
        pred_qa_pairs,
        result_dir=args.result_dir,
        filename=f"test_vqa_result",
        remove_duplicate="qid",
    )
    metrics = report_metrics(result_file=result_file, split='test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiple-Choice Video QA Evaluation Script.')

    parser.add_argument('--model-path', default='')
    parser.add_argument('--result-dir', default='./output/')
    parser.add_argument('--output-dir', default='./output/medical/')
    parser.add_argument('--video-folder', default='../../data/medical/raw_data/Slake1.0/imgs')
    # parser.add_argument('--test-csv', default='../../data/hateful_memes/raw_data/test_seen.jsonl')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)
