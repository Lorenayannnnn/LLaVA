#!/bin/bash

# Get the current date in YYYYMMDD format
#current_date=$(date +"%Y%m%d")
current_date="20250507"
export WANDB_PROJECT=llava
vision_token_attn="causal"
max_train_samples=40000
#shuffle_trivial_vision_tokens_keep_percentage=0.9   # keep top 0.25
#method_name="dropout_by_each_head_each_token_for_txt"   # shuffle_by_CLS, shuffle_by_last_text, dropout_by_last_text_attn_for_txt, dropout_by_last_text_attn_for_all, dropout_by_each_head_each_token_for_txt, dropout_by_each_head_each_token_for_all, dropout_by_nucleus_each_head_each_token_for_all, dropout_by_nucleus_renormalize_each_head_each_token_for_all
#method_name="dropout_by_nucleus_renormalize_each_head_each_token_for_all"
#method_name="dropout_by_nucleus_each_head_each_token_for_txt"
vision_token_pos_enc="none_for_vis_key"   # rope, none, constant_vis_key, constant_vis_qk, none_for_vis_key
#lora_target_modules=("q_proj" "v_proj" "k_proj" "o_proj")     #['o_proj', 'q_proj', 'down_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']
lora_target_modules=("all")
lora_target_modules_str=$(IFS=_; echo "${lora_target_modules[*]}")

#deepspeed --include=localhost:0,1,2,3 --master_port=29501 llava/train/train_mem.py \
deepspeed --master_port=29501 --include=localhost:4,5,7 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --lora_target_modules "${lora_target_modules[@]}" \
    --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_40000_mix.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --max_train_samples $max_train_samples \
    --vision_token_attn $vision_token_attn \
    --vision_token_pos_enc $vision_token_pos_enc \
    --run_name "${current_date}_lora_${lora_target_modules_str}_${max_train_samples}-vis_tok_pos_enc_${vision_token_pos_enc}-vision_${vision_token_attn}-llava-v1.5-7b-lora" \
    --output_dir "./outputs_after_stage_2/lora_checkpoints/${lora_target_modules_str}/${current_date}_train_${max_train_samples}_mix/vis_tok_pos_enc_${vision_token_pos_enc}-vision_${vision_token_attn}-llava-v1.5-7b-lora" \

#bash scripts/my_scripts/after_stage_2_finetune_lora_pos_enc.sh
