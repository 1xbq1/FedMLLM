#!/bin/bash

GPU=0
cd /path/to/code/YOCO/finetune/
CUDA_VISIBLE_DEVICES=$GPU sh finetune_lora.sh > print.txt 2>&1
#cd ..
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_alg.py > print_eval_yogi_5.txt 2>&1 --fed_alg 'fedyogi' --epoch 0
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_alg.py > print_eval_yogi_10.txt 2>&1 --fed_alg 'fedyogi' --epoch 1
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_alg.py > print_eval_adam_5.txt 2>&1 --fed_alg 'fedadam' --epoch 0
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_alg.py > print_eval_adam_10.txt 2>&1 --fed_alg 'fedadam' --epoch 1
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_alg.py > print_eval_avgm_5.txt 2>&1 --fed_alg 'fedavgm' --epoch 0
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_alg.py > print_eval_avgm_10.txt 2>&1 --fed_alg 'fedavgm' --epoch 1
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_alg.py > print_eval_adagrad_5.txt 2>&1 --fed_alg 'fedadagrad' --epoch 0
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_alg.py > print_eval_adagrad_10.txt 2>&1 --fed_alg 'fedadagrad' --epoch 1
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_alg.py > print_eval_avg_5.txt 2>&1 --fed_alg 'fedavg' --epoch 0
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_alg.py > print_eval_avg_10.txt 2>&1 --fed_alg 'fedavg' --epoch 1
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_ensemble.py > print_eval_ens_5.txt 2>&1 --epoch 0
#CUDA_VISIBLE_DEVICES=$GPU python eval_hateful_ensemble.py > print_eval_ens_10.txt 2>&1 --epoch 1

#for c in {5..9}; do
#    for e in {4..9..5}; do
#        CUDA_VISIBLE_DEVICES=$GPU python eval_hateful.py > print_eval_${c}_${e}.txt 2>&1 --model-path client-$c --epoch $e
#    done
#done
