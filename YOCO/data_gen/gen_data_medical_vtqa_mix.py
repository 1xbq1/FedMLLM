import json
import random
import os
import numpy as np

random.seed(41)

num_clients = 10
num_modalities = 2
q_rate = 0.4 #missing rate
p = 1 - q_rate
modality_presence = np.random.binomial(1, p, size=(num_clients, num_modalities))

image_path = f'../../../data/medical/raw_data/vqarad_train.json'
text_path = f'../../../data/medical/raw_data/medalpha_train.json'
output_path = f'../../../data/medical/minicpmv_data/modality-mix-1/qrate-{q_rate}'

id_counter = 0

def process(split_data, image_flag):
    global id_counter

    new_data = []
    for item in split_data:
        sample = {}
        sample['id'] = id_counter
        id_counter += 1
        
        if image_flag:
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
    
    return new_data

def get_splits(path, n):

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total_samples = len(data)

    split_size = total_samples // n
    random.shuffle(data)
    splits = [data[i*split_size:(i+1)*split_size] for i in range(n)]

    if total_samples % n != 0:
        splits[-1].extend(data[n*split_size:])
    
    return splits

def split_data():
    img_n = 0
    txt_n = 0
    for client_id in range(num_clients):
        if (modality_presence[client_id][0] == 1) and (modality_presence[client_id][1] == 1):
            img_n += 1
            txt_n += 1
        elif modality_presence[client_id][0] == 1:
            img_n += 1
        elif modality_presence[client_id][1] == 1:
            txt_n += 1
        else:
            if (random.random() < 0.5):
                modality_presence[client_id][0] = 1
                img_n += 1
            else:
                modality_presence[client_id][1] = 1
                txt_n += 1
    
    img_splits = get_splits(image_path, img_n)
    txt_splits = get_splits(text_path, txt_n)

    file_names = [f'client_{i}.json' for i in range(num_clients)]

    img_vis = 0
    txt_vis = 0
    for client_id, file_name in enumerate(file_names):

        if (modality_presence[client_id][0] == 1) and (modality_presence[client_id][1] == 1):
            img_split_data = img_splits[img_vis]
            img_vis +=1
            img_new_data = process(img_split_data, image_flag=True)

            txt_split_data = txt_splits[txt_vis]
            txt_vis +=1
            txt_new_data = process(txt_split_data, image_flag=False)

            new_data = img_new_data + txt_new_data
            random.shuffle(new_data)

        elif modality_presence[client_id][0] == 1:
            img_split_data = img_splits[img_vis]
            img_vis +=1
            new_data = process(img_split_data, image_flag=True)

        elif modality_presence[client_id][1] == 1:
            txt_split_data = txt_splits[txt_vis]
            txt_vis +=1
            new_data = process(txt_split_data, image_flag=False)

        else:
            print("Error!")
        
        with open(f'{output_path}/{file_name}', 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
        print(f'save in {output_path}/{file_name}')

split_data()
