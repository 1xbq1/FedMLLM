import json
import random
import os

random.seed(41)

num_clients = 10
selected_image_num = 7
selected_image_clients = random.sample(list(range(num_clients)), selected_image_num)

image_path = f'../../../data/medical/raw_data/vqarad_train.json'
text_path = f'../../../data/medical/raw_data/medalpha_train.json'
output_path = f'../../../data/medical/minicpmv_data/modality-single-1/image-{selected_image_num}'

id_counter = 0

def split_data(path, n, client_list, image_vis=False):
    global id_counter
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total_samples = len(data)

    split_size = total_samples // n
    random.shuffle(data)
    splits = [data[i*split_size:(i+1)*split_size] for i in range(n)]

    if total_samples % n != 0:
        splits[-1].extend(data[n*split_size:])

    file_names = [f'client_{i}.json' for i in client_list]

    for file_name, split_data in zip(file_names, splits):
        new_data = []
        for item in split_data:
            sample = {}
            sample['id'] = id_counter
            id_counter += 1
        
            if image_vis:
                sample['image'] = '/path/to/training_data/data/medical/raw_data/images/'+item['image_name']
                sample['conversations'] = [
                    {'role': 'user', 'content': '<image>\n'+str(item['question'])},
                    {'role': 'assistant', 'content': str(item['answer'])}
                ]
            else:
                sample['image'] = None
                sample['conversations'] = [
                    {'role': 'user', 'content': str(item['question'])},
                    {'role': 'assistant', 'content': str(item['answer'])}
                ]
            new_data.append(sample)
        with open(f'{output_path}/{file_name}', 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
        print(f'save in {output_path}/{file_name}')

complete_list = list(range(num_clients))
text_clients_list = list(set(complete_list) - set(selected_image_clients))
split_data(image_path, selected_image_num, selected_image_clients, image_vis=True)
split_data(text_path, num_clients-selected_image_num, text_clients_list)

