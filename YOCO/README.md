# YOCO: You Only Communicate Once

This branch contains the implementation of **YOCO**, a one-shot federated fine-tuning method built upon [FedMLLM](https://github.com/1xbq1/FedMLLM/tree/main).

## 🚀 Introduction
**YOCO** (You Only Communicate Once) is a one-shot Federated Low-Rank Adaptation (LoRA) framework for Multimodal Large Language Models (MLLMs), designed to minimize communication costs while maintaining high performance.

## 🛠️ Installation & Datasets
The environment setup and dataset preparation (Hateful-Memes, CrisisMMD, VQA-RAD, etc.) are consistent with the **FedMLLM** base project. 
Please refer to the [Main Branch README](https://github.com/1xbq1/FedMLLM/blob/main/README.md) for detailed instructions.

## 🏃 Training & Evaluation
To train and test YOCO, simply run the provided script:
```bash
sh start.sh
```

## 📄 Citation
If you find this code useful for your research, please cite our NeurIPS 2025 paper:

```
@inproceedings{xu2025you,
  title={You Only Communicate Once: One-shot Federated Low-Rank Adaptation of {MLLM}},
  author={Binqian Xu and Haiyang Mei and Zechen Bai and Jinjin Gong and Rui Yan and Guo-Sen Xie and Yazhou Yao and Basura Fernando and Xiangbo Shu},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={[https://openreview.net/forum?id=FoVF3iL6o3](https://openreview.net/forum?id=FoVF3iL6o3)}
}

@article{xu2024fedmllm,
  title={FedMLLM: Federated Fine-tuning MLLM on Multimodal Heterogeneity Data},
  author={Xu, Binqian and Shu, Xiangbo and Mei, Haiyang and Xie, Guosen and Fernando, Basura and Tang, Jinhui},
  journal={arXiv preprint arXiv:2411.14717},
  year={2024}
}
```

## 🤝 Acknowledgements
This implementation is built upon FedMLLM, MiniCPM-V, and OpenFedLLM.