cd "/home/ty/code/ty/code/PAA/pekinese/ViCross"

#INPUT_SIZE=224 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen2-vl-7b-instruct \
    --model_id_or_path '/home/ty/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct' \
    --dataset 'pie' \
    --seed 2000 \
    --batch_size 4 \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 10 \
    --system 'You are an expert pedestrian-behavior evaluator.' \
    --dtype 'fp16' \
    --lora_dtype 'fp16' \
    --output_dir '/home/ty/code/ty/code/PAA/pekinese/ViCross/ty/output/pie' \
    --add_output_dir_suffix False \
    --evaluation_strategy 'no' \
    --save_steps 298 \
    --bg_downsample_ratio 0.1 \
    --pan_resolution_h 224 \
    --pan_resolution_w 448 \
    --pan_patch_size 14 \
    --ped_resolution_h 112 \
    --ped_resolution_w 112 \
    --ped_patch_size 7 \
    --lossweight_class 1 \
    --lossweight_pre 1 \
    --lossweight_obs 1 \
    --crop_resize_ratio 1.5 \
    --prompt_length 'medium'






