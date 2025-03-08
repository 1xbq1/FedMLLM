#!/bin/bash

GPU=0

#Hateful-Memes
cd /root/code/FedMLLM/finetune/
CUDA_VISIBLE_DEVICES=$GPU sh finetune_lora.sh > print.txt 2>&1
cd ..
CUDA_VISIBLE_DEVICES=$GPU python eval_hateful.py > print_eval_25.txt 2>&1 --epoch 25

#CrisisMMD
cd /root/code/FedMLLM/finetune/
CUDA_VISIBLE_DEVICES=$GPU sh finetune_lora.sh > print.txt 2>&1
cd ..
CUDA_VISIBLE_DEVICES=$GPU python eval_crisismmd.py > print_eval_40.txt 2>&1 --epoch 40

#MedAlpaca, VQA-RAD, SLAKE
cd /root/code/FedMLLM/finetune/
CUDA_VISIBLE_DEVICES=$GPU sh finetune_lora.sh > print.txt 2>&1
cd ..
CUDA_VISIBLE_DEVICES=$GPU python eval_medical.py --model-path 50
CUDA_VISIBLE_DEVICES=$GPU python eval_medical_slake.py --model-path 50

