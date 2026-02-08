import json
import random
import os
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

num_clients = 10
num_modalities = 2
q_rate = 0.3 #missing rate
p = 1 - q_rate
modality_presence = np.random.binomial(1, p, size=(num_clients, num_modalities))
input_folder = f'../../../data/hateful_memes/raw_data/partition-alpha0.5-clt{num_clients}'
output_folder = f'../../../data/hateful_memes/minicpmv_data/modality-mix/qrate-{q_rate}/partition-alpha0.5-clt{num_clients}'
for file_name in os.listdir(input_folder):
    input_file_path = os.path.join(input_folder, file_name)

    data = []

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for idx, linej in enumerate(f):
            linej = linej.strip()
            line = json.loads(linej)
            data_dict = {}
            data_dict['id'] = line['id']
            client_id = int(file_name[7:-5])
            if (modality_presence[client_id][0] == 1) and (modality_presence[client_id][1] == 1):
                image_flag = True
                text_flag = True
            elif modality_presence[client_id][0] == 1:
                image_flag = True
                text_flag = False
            elif modality_presence[client_id][1] == 1:
                image_flag = False
                text_flag = True
            else:
                if (random.random() < 0.5):
                    image_flag = True
                    text_flag = False
                else:
                    image_flag = False
                    text_flag = True
            if image_flag:
                data_dict['image'] = '/path/to/training_data/data/hateful_memes/raw_data/'+line['img']
            else:
                data_dict['image'] = None
            data_dict['conversations'] = []
            conv_dict = {}
            conv_dict['role'] = 'user'

            a0 = 'not-hateful'
            a1 = 'hateful'
            if text_flag and image_flag:
                text = line['text']
                question = 'Is the content hateful based on the text and image?'
                conv_dict['content'] = f"<image>\nSelect the best answer to the following multiple-choice question based on the text and image.\n{text}\n{question}\nOptions:\n(A) {a0}\n(B) {a1}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
            elif text_flag:
                text = line['text']
                question = 'Is the content hateful based on the text?'
                conv_dict['content'] = f"Select the best answer to the following multiple-choice question based on the text.\n{text}\n{question}\nOptions:\n(A) {a0}\n(B) {a1}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
            elif image_flag:
                question = 'Is the content hateful based on the image?'
                conv_dict['content'] = f"<image>\nSelect the best answer to the following multiple-choice question based on the image.\n{question}\nOptions:\n(A) {a0}\n(B) {a1}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
            else:
                question = 'Is the content hateful?'
                conv_dict['content'] = f"Select the best answer to the following multiple-choice question.\n{question}\nOptions:\n(A) {a0}\n(B) {a1}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
            data_dict["conversations"].append(conv_dict)
            conv_dict = {}
            conv_dict["role"] = "assistant"
            if int(line['label']) == 0:
                conv_dict["content"] = f"(A) {a0}"
            else:
                conv_dict["content"] = f"(B) {a1}"
            data_dict["conversations"].append(conv_dict)
            data.append(data_dict)

    output_file_path = os.path.join(output_folder, file_name)
    with open(output_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print(f'save in {output_file_path}')
