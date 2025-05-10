#!/bin/bash

#MODEL_PATH=outputs/checkpoints/train_10000_mcq/keep_0.25-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mcq/keep_0.5-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mcq/keep_0.1-vision_causal-llava-v1.5-7b-finetune

#MODEL_PATH=outputs/checkpoints/train_10000_mix/vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/keep_0.25-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/keep_0.25-vision_full-llava-v1.5-7b-finetune

#MODEL_PATH=outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_none-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_key-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_key-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_qk-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/vis_tok_pos_enc_constant_vis_qk-vision_causal-llava-v1.5-7b-finetune

#MODEL_PATH=outputs/checkpoints/train_10000_mix/dropout_by_last_text_attn_for_all_keep_0.25-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/dropout_by_each_head_each_token_for_all_keep_0.25-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/dropout_by_each_head_each_token_for_all_keep_0.5-vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_10000_mix/dropout_by_each_head_each_token_for_txt_keep_0.25-vision_full-llava-v1.5-7b-finetune

#MODEL_PATH=outputs/checkpoints/train_40000_mix/vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_40000_mix/vision_full-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_40000_mix/vis_tok_pos_enc_constant_vis_key-vision_causal-llava-v1.5-7b-finetune
#MODEL_PATH=outputs/checkpoints/train_40000_mix/vis_tok_pos_enc_constant_vis_key-vision_full-llava-v1.5-7b-finetune

# LoRA
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/vision_causal-llava-v1.5-7b-lora
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/vision_full-llava-v1.5-7b-lora
# LoRA + attn dropout
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/dropout_by_each_head_each_token_for_txt_keep_0.7_vision_full-llava-v1.5-7b-lora
#MODEL_PATH=outputs/lora_checkpoints/q_proj_v_proj/train_40000_mix/dropout_by_each_head_each_token_for_all_keep_0.7_vision_full-llava-v1.5-7b-lora
# LoRA + attn dropout + p sampling
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
#MODEL_PATH=/home/tianyi/LLaVA/outputs_after_stage_2/lora_checkpoints/all/20250507_train_40000_mix/dropout_by_nucleus_renormalize_each_head_each_token_for_all_keep_0.9_vision_causal-llava-v1.5-7b-lora
#DEVICE=1


#echo "Evaluate ${MODEL_PATH}"
#
##--model-base lmsys/vicuna-7b-v1.5 \
## TODO add model-based and pretrain_mm_mlp_adapter if doing lora
#CUDA_VISIBLE_DEVICES=${DEVICE} python -m llava.eval.model_vqa_vstar \
#    --model-path $MODEL_PATH \
#    --model-base liuhaotian/llava-v1.5-7b \
#    --question-file ./playground/data/vstar_bench/test_questions.jsonl \
#    --image-folder ./playground/data/vstar_bench\
#    --output_dir $MODEL_PATH/vstar_bench \
#    --single-pred-prompt \
#    --temperature 0 \
#    --conv-mode vicuna_v1 \
##    --output_dir outputs/test \


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
    CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_vstar \
      --model-path ${MODEL_PATH} \
      --model-base liuhaotian/llava-v1.5-7b \
      --question-file ./playground/data/vstar_bench/test_questions.jsonl \
      --image-folder ./playground/data/vstar_bench\
      --output_dir ${MODEL_PATH}/w_dropout/vstar_bench \
      --single-pred-prompt \
      --temperature 0 \
      --conv-mode vicuna_v1
done

#bash scripts/v1_5/eval/vstar.sh
