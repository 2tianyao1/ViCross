cd "/home/ty/code/ty/code/PAA/pekinese/ViCross/ty"

checkpoints=(1340)
for i in "${checkpoints[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python infer.py \
      --seed 2000 \
      --dataset jaad_beh \
      --model_id_or_path '/home/ty/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct' \
      --ckpt_dir /home/ty/code/ty/code/PAA/pekinese/PAALLM_V4_3_ablation_1/ty/output/ablation/seed/2000/checkpoint-${i} \
      --metric_output_file '/home/ty/code/ty/code/PAA/pekinese/PAALLM_V4_3_ablation_1/ty/output/ablation/seed/2000/jaad_beh.xlsx' \
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
      --prompt_length 'short' \
      --system 'You are an autonomous driving system.'
done
