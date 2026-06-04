import json
import random
import os
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

# ============================================================
# Switch this to one of: 'P1', 'P2', 'P3', 'P4', 'P5'
PROMPT_ID = 'P5'
# Hybrid-modal scenario: each client has each modality with probability p
# p = 0.7  (i.e., q_rate = 1 - p = 0.3, matches the paper's default mix setting)
num_clients = 10
num_modalities = 2
q_rate = 0.3                    # missing rate at client level
p = 1 - q_rate                  # presence rate = 0.7
# ============================================================

PROMPTS = {
    # P1: Original Prompt
    'P1': {
        'preamble': 'Select the best answer to the following multiple-choice question based on the text and image.',
        'question': 'Is the content hateful based on the text and image?',
    },
    # P2: Relaxed Modality-Specific Prompt
    'P2': {
        'preamble': 'Select the best answer to the following multiple-choice question based on the available content from the text and image.',
        'question': 'Is the content hateful based on the available content from the text and image?',
    },
    # P3: Our Prompt (== existing _aug version)
    'P3': {
        'preamble': 'Select the best answer to the following multiple-choice question, without considering the modality.',
        'question': 'Is the content hateful, without considering the modality?',
    },
    # P4: Strict Modality-Agnostic Prompt
    'P4': {
        'preamble': 'Select the best answer to the following multiple-choice question based on all available information, regardless of its modality.',
        'question': 'Is the content hateful based on all available information, regardless of its modality?',
    },
    # P5: Over-Relaxed Prompt
    'P5': {
        'preamble': 'Select the best answer to the following multiple-choice question.',
        'question': 'Is the content hateful?',
    },
}

assert PROMPT_ID in PROMPTS, f"Unknown PROMPT_ID={PROMPT_ID}; choose from {list(PROMPTS)}"
preamble = PROMPTS[PROMPT_ID]['preamble']
question = PROMPTS[PROMPT_ID]['question']

# Decide each client's modality presence ONCE (client-level, not sample-level)
# modality_presence[i] = [has_image, has_text] for client i
modality_presence = np.random.binomial(1, p, size=(num_clients, num_modalities))

input_folder = f'../../../data/hateful_memes/raw_data/partition-alpha0.5-clt{num_clients}'
output_folder = (
    f'../../../data/hateful_memes/minicpmv_data/'
    f'modality-mix-{PROMPT_ID}/qrate-{q_rate}/partition-alpha0.5-clt{num_clients}'
)
os.makedirs(output_folder, exist_ok=True)
print(f'[Prompt {PROMPT_ID}] writing to {output_folder}')
print(f'modality_presence (rows=clients, cols=[image,text]):\n{modality_presence}')

for file_name in os.listdir(input_folder):
    input_file_path = os.path.join(input_folder, file_name)

    data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for idx, linej in enumerate(f):
            linej = linej.strip()
            line = json.loads(linej)
            data_dict = {}
            data_dict['id'] = line['id']
            client_id = int(file_name[7:-5])  # client_{i}.json → i

            # Decide which modalities this sample has, based on the client's modality set
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
                # client has neither modality → fall back to a random single modality per sample
                if random.random() < 0.5:
                    image_flag = True
                    text_flag = False
                else:
                    image_flag = False
                    text_flag = True

            if image_flag:
                data_dict['image'] = '/path/to/training_data/data/hateful_memes/raw_data/' + line['img']
            else:
                data_dict['image'] = None

            data_dict['conversations'] = []
            conv_dict = {'role': 'user'}

            a0 = 'not-hateful'
            a1 = 'hateful'

            # Same prompt across all 4 branches; only the surrounding image / text content changes
            image_prefix = '<image>\n' if image_flag else ''
            text_line = f"{line['text']}\n" if text_flag else ''

            conv_dict['content'] = (
                f"{image_prefix}{preamble}\n{text_line}{question}\n"
                f"Options:\n(A) {a0}\n(B) {a1}\n"
                f"Answer with the option's letter from the given choices directly and only give the best option. The best answer is: "
            )
            data_dict['conversations'].append(conv_dict)

            conv_dict = {'role': 'assistant'}
            if int(line['label']) == 0:
                conv_dict['content'] = f"(A) {a0}"
            else:
                conv_dict['content'] = f"(B) {a1}"
            data_dict['conversations'].append(conv_dict)

            data.append(data_dict)

    output_file_path = os.path.join(output_folder, file_name)
    with open(output_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f'save in {output_file_path}')
