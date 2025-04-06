#!/bin/bash

#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mcq/keep_-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mcq/keep_-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mcq/keep_0.25-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mcq/keep_0.5-vision_causal-llava-v1.5-7b-finetune
MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mcq/keep_0.1-vision_causal-llava-v1.5-7b-finetune

CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file $MODEL_PATH/pope/preds.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --vision_token_attn causal
    # TODO change vision_token_attn

echo "Start eval ${MODEL_PATH}/pope/preds.jsonl"
python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $MODEL_PATH/pope/preds.jsonl \

#bash scripts/v1_5/eval/pope.sh
