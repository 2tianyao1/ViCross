cd "/home/ty/code/ty/code/PAA/pekinese/ViCross/ty"

checkpoints=(1340)
for i in "${checkpoints[@]}"
do
  CUDA_VISIBLE_DEVICES=2 python infer.py \
      --dataset pie \
      --ckpt_dir /home/ty/code/ty/code/PAA/pekinese/PAALLM_V4_3/ty/output/base/pie/checkpoint-${i} \
      --metric_output_file '/home/ty/code/ty/code/PAA/pekinese/PAALLM_V4_3/ty/output/base/pie/pie.xlsx' \
      --bg_downsample_ratio \
      --pan_resolution_h 112 \
      --pan_resolution_w 224 \
      --pan_patch_size 14 \
      --ped_resolution_h 112 \
      --ped_resolution_w 112 \
      --ped_patch_size 7 \
      --crop_resize_ratio 1.5
done
