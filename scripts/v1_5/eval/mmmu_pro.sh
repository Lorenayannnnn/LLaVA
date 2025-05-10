#!/bin/bash

# 10k data: dropout
#MODEL_PATH=outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_none-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_key-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/dropout_by_last_text_attn_for_all_keep_0.25-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/dropout_by_each_head_each_token_for_all_keep_0.25-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/dropout_by_each_head_each_token_for_all_keep_0.5-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/dropout_by_each_head_each_token_for_txt_keep_0.25-vision_full-llava-v1.5-7b-finetune

#MODEL_PATH=outputs/checkpoints/train_10000_mix/vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_qk-vision_causal-llava-v1.5-7b-finetune    # This is full attention
#MODEL_PATH=outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_key-vision_full-llava-v1.5-7b-finetune

# 40k data
#MODEL_PATH=outputs/checkpoints/train_40000_mix/vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_40000_mix/vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_40000_mix/vis_tok_pos_enc_constant_vis_key-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_40000_mix/vis_tok_pos_enc_constant_vis_key-vision_full-llava-v1.5-7b-finetune

# 40k lora q_proj, k_proj
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/vision_full-llava-v1.5-7b-lora
# attn dropout
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/dropout_by_each_head_each_token_for_txt_keep_0.7_vision_full-llava-v1.5-7b-lora
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/dropout_by_each_head_each_token_for_all_keep_0.7_vision_full-llava-v1.5-7b-lora
# attn dropout + p-sampling
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/dropout_by_nucleus_each_head_each_token_for_all_keep_0.9_vision_full-llava-v1.5-7b-lora
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/dropout_by_nucleus_each_head_each_token_for_txt_keep_0.9_vision_full-llava-v1.5-7b-lora

#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj_k_proj_o_proj/20250428_train_40000_mix/vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj_k_proj_o_proj/20250428_train_40000_mix/vision_full-llava-v1.5-7b-lora
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj_k_proj_o_proj/20250428_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_all_keep_0.9_vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj_k_proj_o_proj/20250428_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_txt_keep_0.9_vision_causal-llava-v1.5-7b-lora

# After stage 2
#MODEL_PATH=outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/vision_full-llava-v1.5-7b-lora
#MODEL_PATH=outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_txt_keep_0.7_vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_txt_keep_0.9_vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_all_keep_0.7_vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_all_keep_0.5_vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/vis_tok_pos_enc_none_for_vis_key-vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_all_keep_0.9_vision_causal-llava-v1.5-7b-lora
#DEVICE=4
#
#echo "Evaluate ${MODEL_PATH}"
#
##--model-base lmsys/vicuna-7b-v1.5 \
#CUDA_VISIBLE_DEVICES=${DEVICE} python -m llava.eval.model_mmmu_pro \
#    --model-path $MODEL_PATH \
#    --model-base liuhaotian/llava-v1.5-7b \
#    --output_dir $MODEL_PATH/mmmu_pro \
#    --single-pred-prompt \
#    --temperature 0 \
#    --conv-mode vicuna_v1 \
#    --skip_multi_images \

ALL_MODEL_PATHS=(
"outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/vision_causal-llava-v1.5-7b-lora"
"outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/vision_full-llava-v1.5-7b-lora"
"outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_all_keep_0.5_vision_causal-llava-v1.5-7b-lora"
"outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_all_keep_0.7_vision_causal-llava-v1.5-7b-lora"
"outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_all_keep_0.9_vision_causal-llava-v1.5-7b-lora"
"outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_txt_keep_0.7_vision_causal-llava-v1.5-7b-lora"
"outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_txt_keep_0.9_vision_causal-llava-v1.5-7b-lora"
"outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/vis_tok_pos_enc_none_for_vis_key-vision_causal-llava-v1.5-7b-lora"
)

for MODEL_PATH in "${ALL_MODEL_PATHS[@]}"; do
    echo "Evaluate ${MODEL_PATH}"
    CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_mmmu_pro \
        --model-path ${MODEL_PATH} \
        --model-base liuhaotian/llava-v1.5-7b \
        --output_dir ${MODEL_PATH}/w_dropout/mmmu_pro \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --skip_multi_images
done


#bash scripts/v1_5/eval/mmmu_pro.sh
