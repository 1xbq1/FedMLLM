#!/bin/bash

# GPU=0

# #Hateful-Memes
# cd /data1_sata/binqian/code/FedMLLM/finetune/
# CUDA_VISIBLE_DEVICES=$GPU sh finetune_lora.sh > print.txt 2>&1
# cd ..
# CUDA_VISIBLE_DEVICES=$GPU python eval_hateful.py > print_eval_25.txt 2>&1 --epoch 25

#CrisisMMD
# cd /root/code/FedMLLM/finetune/
# CUDA_VISIBLE_DEVICES=$GPU sh finetune_lora.sh > print.txt 2>&1
# cd ..
# CUDA_VISIBLE_DEVICES=$GPU python eval_crisismmd.py > print_eval_40.txt 2>&1 --epoch 40

# #MedAlpaca, VQA-RAD, SLAKE
# cd /root/code/FedMLLM/finetune/
# CUDA_VISIBLE_DEVICES=$GPU sh finetune_lora.sh > print.txt 2>&1
# cd ..
# CUDA_VISIBLE_DEVICES=$GPU python eval_medical.py --model-path 50
# CUDA_VISIBLE_DEVICES=$GPU python eval_medical_slake.py --model-path 50

cd /data1_sata/binqian/code/FedMLLM/finetune
GPU=0

for P in P1 P2 P3 P4 P5; do
  echo "==================== Training $P ===================="
  CUDA_VISIBLE_DEVICES=$GPU PROMPT_ID=$P QRATE=$QRATE \
    bash finetune_lora.sh > ../output/print_train_${P}.txt 2>&1

  echo "==================== Eval $P ===================="
  cd /data1_sata/binqian/code/FedMLLM/
  # 默认每 5 round 存一次 checkpoint，num_rounds=50 → 用 checkpoint-25 或 50
  # 这里假设你测 epoch=25；想测最后一个就改成 50
  CKPT_DIR="./output/output_lora_${P}_mix_q${QRATE}/checkpoint-25"
  CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_prompt.py \
    --prompt-id $P \
    --epoch 25 \
    --video-folder /data1_sata/binqian/data/hateful_memes/raw_data \
    --test-csv /data1_sata/binqian/data/hateful_memes/raw_data/test_seen.jsonl \
    > ./output/print_eval_${P}.txt 2>&1

  cd /data1_sata/binqian/code/FedMLLM/finetune
done
