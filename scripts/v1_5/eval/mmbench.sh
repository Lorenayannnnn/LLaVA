#!/bin/bash

SPLIT="mmbench_dev_20230712"
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mcq/keep_-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mcq/keep_-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mcq/keep_0.25-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mcq/keep_0.5-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mcq/keep_0.1-vision_causal-llava-v1.5-7b-finetune

#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/keep_0.25-vision_causal-llava-v1.5-7b-finetune
MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/keep_0.25-vision_full-llava-v1.5-7b-finetune

echo "Evaluate ${MODEL_PATH}"

CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file $MODEL_PATH/mmbench/preds.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
#    --vision_token_attn full
#    TODO change vision_token_attn

#    --answers-file outputs/test/test.jsonl \
#mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT
#
#python scripts/convert_mmbench_for_submission.py \
#    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
#    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
#    --experiment llava-v1.5-13b

#bash scripts/v1_5/eval/mmbench.sh
