## FedMLLM: Federated Fine-tuning MLLM on Multimodal Heterogeneity Data 
[pdf](https://arxiv.org/pdf/2411.14717)

<p align="center">
  <img src="https://github.com/1xbq1/FedMLLM/blob/main/assets/FedMLLM.PNG" width="80%"/>
</p>


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
        │   ├── data_process_medalpaca.py
        │   ├── data_process_vqarad.py
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
        │   ├── gen_data_hateful.py
        │   ├── gen_data_medical_vtqa_single.py
        │   └── gen_data_medical_vtqa_mix.py
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
        ├── eval_hateful.py
        ├── eval_medical_gpt_slake.py
        ├── eval_medical_gpt.py
        ├── eval_medical_slake.py
        ├── eval_medical.py
        ├── vqa_eval_slake.py
        ├── vqa_eval.py
        ├── vqa_slake.py
        ├── vqa.py
        └── start.sh
```

## Install
```Shell
conda create -n FedMLLM python=3.10 -y
pip install -r requirements.txt
```

## Dataset
### Download dataset
1. Hateful-Memes
[download](https://www.kaggle.com/datasets/williamberrios/hateful-memes)
2. CrisisMMD
[download](https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz)
3. VQA-RAD
[download](https://www.kaggle.com/datasets/shashankshekhar1205/vqa-rad-visual-question-answering-radiology)
4. MedAlpaca
[download](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
5. SLAKE
[download](https://www.med-vqa.com/slake/)
### Dataset processing
```bash
cd data_gen/

python data_partition_crisismmd.py
python gen_data_crisismmd.py # aligned modal scenario
python gen_data_crisismmd_missing.py # missing modal scenario
python gen_data_crisismmd_missing_aug.py # missing modal scenario with prompt strategy
python gen_data_crisismmd_single.py # cross modal scenario
python gen_data_crisismmd_mix.py # hybrid modal scenario

python data_process_medalpaca.py
python data_process_vqarad.py
python gen_data_medical_vtqa_mix.py
python gen_data_medical_vtqa_single.py
```

## Training and Testing
```
sh start.sh
```

## Citation
```
@article{xu2024fedmllm,
  title={FedMLLM: Federated Fine-tuning MLLM on Multimodal Heterogeneity Data},
  author={Xu, Binqian and Shu, Xiangbo and Mei, Haiyang and Xie, Guosen and Fernando, Basura and Tang, Jinhui},
  journal={arXiv preprint arXiv:2411.14717},
  year={2024}
}
```

## Acknowledgements
This repo is based on [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V), [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM), and [PeFoMed](https://github.com/jinlHe/PeFoMed/tree/main) thanks to the original authors for their works!
