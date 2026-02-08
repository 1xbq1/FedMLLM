import json

with open('/path/to/training_data/data/medical/raw_data/medical_meadow_wikidoc_medical_flashcards.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

modified_data = []

for item in data:
    if 'input' in item and 'output' in item:
        modified_item = {
            'question': item['input'],
            'answer': item['output']
        }
        modified_data.append(modified_item)

with open('/path/to/training_data/data/medical/raw_data/medalpha_train.json', 'w', encoding='utf-8') as f:
    json.dump(modified_data, f, ensure_ascii=False, indent=4)



