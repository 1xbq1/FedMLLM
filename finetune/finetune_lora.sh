#!/bin/bash

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="openbmb/MiniCPM-V-2_6-int4"
#MODEL="openbmb/MiniCPM-Llama3-V-2_5"
PROMPT_ID="${PROMPT_ID:-P3}"           # 从环境变量读
QRATE="${QRATE:-0.3}"
DATA="/data1_sata/binqian/data/hateful_memes/minicpmv_data/modality-mix-${PROMPT_ID}/qrate-${QRATE}/partition-alpha0.5-clt10"
LLM_TYPE="qwen2"
#LLM_TYPE="llama3"
MODEL_MAX_Length=1200

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision false \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --num_train_epoch 1 \
    --early_stop_round 25 \
    --eval_steps 50000 \
    --output_dir ../output/output_lora_${PROMPT_ID}_mix_q${QRATE} \
    --logging_dir ../output/output_lora_${PROMPT_ID}_mix_q${QRATE} \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --init_learning_rate 2e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --report_to "tensorboard" # wandb
