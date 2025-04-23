#!/bin/bash

# 10k data: dropout
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_none-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_key-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/dropout_by_last_text_attn_for_all_keep_0.25-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/dropout_by_each_head_each_token_for_all_keep_0.25-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/dropout_by_each_head_each_token_for_all_keep_0.5-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/dropout_by_each_head_each_token_for_txt_keep_0.25-vision_full-llava-v1.5-7b-finetune

#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_qk-vision_causal-llava-v1.5-7b-finetune    # This is full attention
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_key-vision_full-llava-v1.5-7b-finetune

# 40k data
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_40000_mix/vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_40000_mix/vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_40000_mix/vis_tok_pos_enc_constant_vis_key-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=/home/tianyi/LLaVA/outputs/checkpoints/train_40000_mix/vis_tok_pos_enc_constant_vis_key-vision_full-llava-v1.5-7b-finetune

# 40k lora q_proj, k_proj
#MODEL_PATH=/home/tianyi/LLaVA/outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=/home/tianyi/LLaVA/outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/vision_full-llava-v1.5-7b-lora
# attn dropout
#MODEL_PATH=/home/tianyi/LLaVA/outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/dropout_by_each_head_each_token_for_txt_keep_0.7_vision_full-llava-v1.5-7b-lora
#MODEL_PATH=/home/tianyi/LLaVA/outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/dropout_by_each_head_each_token_for_all_keep_0.7_vision_full-llava-v1.5-7b-lora
# attn dropout + p-sampling
#MODEL_PATH=/home/tianyi/LLaVA/outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/dropout_by_nucleus_each_head_each_token_for_all_keep_0.9_vision_full-llava-v1.5-7b-lora
MODEL_PATH=/home/tianyi/LLaVA/outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/dropout_by_nucleus_each_head_each_token_for_txt_keep_0.9_vision_full-llava-v1.5-7b-lora
#DEVICE=4,5
DEVICE=6,7


echo "Evaluate ${MODEL_PATH}"

CUDA_VISIBLE_DEVICES=${DEVICE} python -m llava.eval.model_mmmu_pro \
    --model-path $MODEL_PATH \
    --model-base lmsys/vicuna-7b-v1.5 \
    --output_dir $MODEL_PATH/mmmu_pro \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --skip_multi_images \

#bash scripts/v1_5/eval/mmmu_pro.sh
