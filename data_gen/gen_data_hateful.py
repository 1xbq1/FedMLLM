import json
import random
import os

input_folder = '../../../data/hateful_memes/raw_data/partition-alpha0.5-clt10'
output_folder = '../../../data/hateful_memes/minicpmv_data/partition-alpha0.5-clt10'
for file_name in os.listdir(input_folder):
    input_file_path = os.path.join(input_folder, file_name)

    data = []

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for idx, linej in enumerate(f):
            linej = linej.strip()
            line = json.loads(linej)
            data_dict = {}
            data_dict['id'] = line['id']
            data_dict['image'] = '/path/to/training_data/data/hateful_memes/raw_data/'+line['img']
            data_dict['conversations'] = []
            conv_dict = {}
            conv_dict['role'] = 'user'

            a0 = 'not-hateful'
            a1 = 'hateful'
            text = line['text']
            question = 'Is the content hateful based on the text and image?'
            conv_dict['content'] = f"<image>\nSelect the best answer to the following multiple-choice question based on the text and image.\n{text}\n{question}\nOptions:\n(A) {a0}\n(B) {a1}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
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
