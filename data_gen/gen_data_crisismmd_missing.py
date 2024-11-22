import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import os

def remove_url(text):
    text = re.sub(r'http\S+', '', text)
    return(text)

random.seed(42)
missing_rate = 0.5
input_folder = '../../../data/crisis-mmd/raw_data/partition-alpha0.5-clt10'
output_folder = f'../../../data/crisis-mmd/minicpmv_data/modality-missing/mrate-{missing_rate}/partition-alpha0.5-clt10'
for file_name in os.listdir(input_folder):
    input_file_path = os.path.join(input_folder, file_name)

    data = []

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for idx, linej in enumerate(f):
            linej = linej.strip()
            line = json.loads(linej)
            data_dict = {}
    
            data_dict['id'] = line['image_id']

            text_flag = (random.random() < (1-missing_rate))
            image_flag = (random.random() < (1-missing_rate))

            if image_flag:
                data_dict['image'] = '/path/to/training_data/data/crisis-mmd/raw_data/'+line['image']
            else:
                data_dict['image'] = None
            data_dict['conversations'] = []
            conv_dict = {}
            conv_dict['role'] = 'user'

            ax = ['affected_individuals',
                  'infrastructure_and_utility_damage',
                  'injured_or_dead_people',
                  'missing_or_found_people',
                  'rescue_volunteering_or_donation_effort',
                  'vehicle_damage',
                  'other_relevant_information',
                  'not_humanitarian']

            text = remove_url(line['tweet_text']).strip()
            options = ''
            for idx, ax_idx in enumerate(ax):
                options += '\n(' + chr(ord('A') + idx) + ') ' + ax_idx
                # print(options)
            if text_flag and image_flag:
                question = 'What is the humanitarian category based on the image and text?'
                conv_dict['content'] = f"<image>\nSelect the best answer to the following multiple-choice question based on the text and image.\n{text}\n{question}\nOptions:{options}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
            elif text_flag:
                question = 'What is the humanitarian category based on the text?'
                conv_dict['content'] = f"Select the best answer to the following multiple-choice question based on the text.\n{text}\n{question}\nOptions:{options}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
            elif image_flag:
                question = 'What is the humanitarian category based on the image?'
                conv_dict['content'] = f"<image>\nSelect the best answer to the following multiple-choice question based on the image.\n{question}\nOptions:{options}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
            else:
                question = 'What is the humanitarian category?'
                conv_dict['content'] = f"Select the best answer to the following multiple-choice question.\n{question}\nOptions:{options}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
            
            data_dict["conversations"].append(conv_dict)
            conv_dict = {}
            conv_dict["role"] = "assistant"
            label_image = line['label_image']
            for idx, ax_idx in enumerate(ax):
                if label_image == ax_idx:
                    conv_dict["content"] = '(' + chr(ord('A') + idx) + ') ' + ax_idx
                    break
            data_dict["conversations"].append(conv_dict)
            data.append(data_dict)

    output_file_path = os.path.join(output_folder, file_name)
    with open(output_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print(f'save in {output_file_path}')
