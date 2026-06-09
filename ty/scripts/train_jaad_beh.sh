cd "/home/ty/code/ty/code/PAA/pekinese/ViCross"

#INPUT_SIZE=224 \
CUDA_VISIBLE_DEVICES=1 \
swift sft \
    --model_type qwen2-vl-7b-instruct \
    --model_id_or_path '/home/ty/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct' \
    --dataset 'jaad_beh' \
    --seed 2030 \
    --batch_size 4 \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 10 \
    --system 'You are an autonomous driving system.' \
    --dtype 'fp16' \
    --sft_type 'lora' \
    --lora_dtype 'fp16' \
    --output_dir '//home/ty/code/ty/code/PAA/pekinese/ViCross/ty/output/jaad_beh' \
    --add_output_dir_suffix False \
    --evaluation_strategy 'no' \
    --save_steps 133 \
    --bg_downsample_ratio 0.5 \
    --pan_resolution_h 224 \
    --pan_resolution_w 448 \
    --pan_patch_size 14 \
    --ped_resolution_h 112 \
    --ped_resolution_w 112 \
    --ped_patch_size 7 \
    --lossweight_class 0.5 \
    --lossweight_pre 0.5 \
    --lossweight_obs 0.5 \
    --crop_resize_ratio 1.5 \
    --prompt_length 'short'







