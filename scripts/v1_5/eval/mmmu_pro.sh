#!/bin/bash


#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_none-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_key-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_qk-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_qk-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_key-vision_full-llava-v1.5-7b-finetune

#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_40000_mix/vision_causal-llava-v1.5-7b-finetune
#DEVICE=4,5
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_40000_mix/vision_full-llava-v1.5-7b-finetune
#DEVICE=6,7
MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_40000_mix/vis_tok_pos_enc_constant_vis_key-vision_causal-llava-v1.5-7b-finetune
DEVICE=4,5
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_40000_mix/vis_tok_pos_enc_constant_vis_key-vision_full-llava-v1.5-7b-finetune
#DEVICE=3

echo "Evaluate ${MODEL_PATH}"

CUDA_VISIBLE_DEVICES=${DEVICE} python -m llava.eval.model_mmmu_pro \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/vstar_bench/test_questions.jsonl \
    --image-folder ./playground/data/vstar_bench\
    --output_dir $MODEL_PATH/mmmu_pro \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --skip_multi_images \

#bash scripts/v1_5/eval/mmmu_pro.sh
