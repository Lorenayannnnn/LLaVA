#!/bin/bash

export WANDB_PROJECT=llava
vision_token_attn="causal"
max_train_samples=10000
shuffle_trivial_vision_tokens_keep_percentage=0.25   # keep top 0.25
#deepspeed --include=localhost:0,1,2,3 llava/train/train_mem.py \
deepspeed --master_port=29501 --include=localhost:4,5,6,7 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_10k_mix.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter outputs/pretrain_checkpoints/liuhaotian/llava-v1.5-7b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
    --vision_token_attn $vision_token_attn \
    --shuffle_trivial_vision_tokens_keep_percentage $shuffle_trivial_vision_tokens_keep_percentage \
    --output_dir "./outputs/checkpoints/train_${max_train_samples}_mix/keep_${shuffle_trivial_vision_tokens_keep_percentage}-vision_${vision_token_attn}-llava-v1.5-7b-finetune" \
    --run_name "train_${max_train_samples}-keep_${shuffle_trivial_vision_tokens_keep_percentage}_mcq-vision_${vision_token_attn}-llava-v1.5-7b-finetune" \
    --max_train_samples $max_train_samples \

#    --output_dir "./outputs/checkpoints/train_${max_train_samples}_mix/vision_${vision_token_attn}-llava-v1.5-7b-finetune" \
#    --run_name "train_${max_train_samples}_mix-vision_${vision_token_attn}-llava-v1.5-7b-finetune" \

#    --output_dir "./outputs/test" \
#bash scripts/v1_5/finetune.sh
