import json

with open("/path/to/training_data/data/medical/raw_data/VQA_RAD Dataset Public.json", "r") as f:
    data = json.load(f)

train_data = []
test_data = []

for sample in data:
    phrase_type = sample.get('phrase_type')

    if phrase_type in ["freeform", "para"]:
        filtered_sample = {
            'image_name': sample.get('image_name'),
            'question': sample.get('question'),
            'answer': sample.get('answer')
        }
        train_data.append(filtered_sample)
    elif phrase_type in ["test_freeform", "test_para"]:
        test_data.append(sample)

with open("/path/to/training_data/data/medical/raw_data/vqarad_train.json", "w") as f_train:
   json.dump(train_data, f_train, indent=4)

with open("/path/to/training_data/data/medical/raw_data/vqarad_test.json", "w") as f_test:
    json.dump(test_data, f_test, indent=4)

# return train_data, test_data
