#!/bin/bash
datasets_jsonl='/XXX/Medal-S/data/challenge_data/train_all.jsonl'
target_spacing=(1.0 1.0 1.0)
crop_size=(192 192 192)
echo "Start preprocessing data, target spacing: ${target_spacing[@]}, crop size: ${crop_size[@]}"
python \
/XXX/Medal-S/data/preprocess.py \
--datasets_jsonl ${datasets_jsonl} \
--target_spacing ${target_spacing[@]} \
--crop_size ${crop_size[@]} \
--num_workers 32 \

crop_str=$(IFS=_ ; echo "${crop_size[*]}")
spacing_str=$(IFS=_ ; echo "${target_spacing[*]}")
name="Medal_S_CVPR2025_crop_size_${crop_str}_spacing_${spacing_str}_w_simulated_prompt_pretrain_nano_cvpr25_v0"

echo "Start training"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1 #1

torchrun \
--nnodes 1 \
--nproc_per_node 4 \
--master_port 29800 \
/XXX/Medal-S/train.py \
--log_dir '/data/shipengcheng/code/CVPR2025_Text_guided_seg/log' \
--name $name \
--vision_backbone 'UNET' \
--input_channels 2 \
--deep_supervision True \
--save_large_interval 5000 \
--save_small_interval 100 \
--log_step_interval 100 \ 
--step_num 40000 30000 20000 10000 2000 \ 
--warmup 2000 2000 2000 1000 200 \
--lr 1e-4 4e-5 2e-5 2e-5 1e-5 \
--accumulate_grad_interval 1 \
--datasets_jsonl ${datasets_jsonl} \
--text_prompts_json '/XXX/Medal-S/data/challenge_data/CVPR25_TextSegFMData_with_class.json' \
--text_encoder 'ours' \
--text_encoder_checkpoint '/data/shipengcheng/code/CVPR2025_Text_guided_seg/checkpoints/text_encoder_cvpr25_v0.pth' \
--text_encoder_partial_load True \
--open_bert_layer 12 \
--open_modality_embed False \
--dataset_config '/XXX/Medal-S/data/dataset_config/cvpr25.json' \
--num_workers 12 \
--max_queries 32 \
--crop_size ${crop_size[@]} \
--target_spacing ${target_spacing[@]} \
--patch_size 32 32 32 \
--batchsize_3d 2  \
--allow_repeat True \
--pin_memory True \
--nnUNet_aug True \
--resume True