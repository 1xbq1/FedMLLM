## FedMLLM: Federated Fine-tuning MLLM on Multimodal Heterogeneity Data [pdf](https://arxiv.org/pdf/2411.14717)
![image]()

## Directory Structure

```plaintext
.
└── root/
    ├── data/
    │   ├── hateful_memes/
    │   │   ├── minicpmv_data/
    │   │   │   ├── modality-missing/
    │   │   │   │   ├── mrate-0.3/
    │   │   │   │   │   └── partition-alpha0.5-clt10
    │   │   │   │   ├── mrate-0.4/
    │   │   │   │   │   └── partition-alpha0.5-clt10
    │   │   │   │   └── mrate-0.5/
    │   │   │   │       └── partition-alpha0.5-clt10
    │   │   │   ├── modality-single/
    │   │   │   │   ├── image-3/
    │   │   │   │   │   └── partition-alpha0.5-clt10
    │   │   │   │   ├── image-5/
    │   │   │   │   │   └── partition-alpha0.5-clt10
    │   │   │   │   └── image-7/
    │   │   │   │       └── partition-alpha0.5-clt10
    │   │   │   ├── modality-mix/
    │   │   │   │   ├── qrate-0.2/
    │   │   │   │   │   └── partition-alpha0.5-clt10
    │   │   │   │   ├── qrate-0.3/
    │   │   │   │   │   └── partition-alpha0.5-clt10
    │   │   │   │   └── qrate-0.4/
    │   │   │   │       └── partition-alpha0.5-clt10
    │   │   │   ├── partition-alpha5.0-clt10
    │   │   │   ├── partition-alpha1.0-clt10
    │   │   │   └── partition-alpha0.5-clt10
    │   │   └── raw_data/ # Extracted files of the downloaded dataset
    │   │       ├── partition-alpha5.0-clt10
    │   │       ├── partition-alpha1.0-clt10
    │   │       └── partition-alpha0.5-clt10
    │   └── crisis-mmd # Consistent with the *hateful_memes* folder structure.
    └── code/
        ├── data_gen/
        │   ├── data_partition_crisismmd.py
        │   ├── data_partition_hateful.py
        │   ├── gen_data_crisismmd_missing_aug.py
        │   ├── gen_data_crisismmd_missing.py
        │   ├── gen_data_crisismmd_mix_aug.py
        │   ├── gen_data_crisismmd_mix.py
        │   ├── gen_data_crisismmd_single_aug.py
        │   ├── gen_data_crisismmd_single.py
        │   ├── gen_data_crisismmd.py
        │   ├── gen_data_hateful_missing_aug.py
        │   ├── gen_data_hateful_missing.py
        │   ├── gen_data_hateful_mix_aug.py
        │   ├── gen_data_hateful_mix.py
        │   ├── gen_data_hateful_single_aug.py
        │   ├── gen_data_hateful_single.py
        │   └── gen_data_hateful.py
        ├── finetune/
        │   ├── federated_learning/
        │   │   ├── __init__.py
        │   │   ├── fed_global.py
        │   │   └── fed_utils.py
        │   ├── __init__.py
        │   ├── dataset.py
        │   ├── finetune_lora.sh
        │   ├── finetune.py
        │   └── trainer.py
        ├── eval_crisismmd_aug.py
        ├── eval_crisismmd.py
        ├── eval_hateful_aug.py
        └── eval_hateful.py
```

## Install
```Shell
conda create -n FedMLLM python=3.10 -y
pip install -r requirements.txt
pip install deepspeed
pip3 install -U scikit-learn
pip install peft
pip install flash_attn
pip install bitsandbytes
pip install tensorboardX
```

## Dataset
### Download dataset
1. Hateful-Memes
[download](https://www.kaggle.com/datasets/williamberrios/hateful-memes)
2. CrisisMMD
[download](https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz)
### Dataset processing
```bash
python data_partition_crisismmd.py
python gen_data_crisismmd.py # aligned modal scenario
python gen_data_crisismmd_missing.py # missing modal scenario
python gen_data_crisismmd_missing_aug.py # missing modal scenario with prompt strategy
python gen_data_crisismmd_single.py # cross modal scenario
python gen_data_crisismmd_mix.py # hybrid modal scenario
```

## Training
```
cd finetune/
sh finetune_lora.sh
```

## Testing
```
python eval_crisismmd.py
python eval_hateful.py
```

## Acknowledgements
This repo is based on [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) and [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM), thanks to the original authors for their works!
