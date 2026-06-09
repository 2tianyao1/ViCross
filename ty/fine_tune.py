# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch

from swift.llm import (
    InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main
)

# model_type = ModelType.qwen2_vl_7b_instruct
model_type = ModelType.qwen2_vl_7b_instruct
sft_args = SftArguments(
    model_type = model_type,
    model_id_or_path = '/home/ty/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct',
    # dataset=[f'{DatasetName.coco_en_2_mini}#2000'],
    output_dir='/home/ty/code/ty/code/PAA/pekinese/PAALLM_V4_2/ty/output/TEM',
    add_output_dir_suffix=False,
    dtype='fp16',
    lora_dtype='fp16',
    system = 'You are an expert in pedestrian crossing-behavior analysis.',
    eval_strategy = 'no'
    )
result = sft_main(sft_args)
last_model_checkpoint = result['last_model_checkpoint']
print(f'last_model_checkpoint: {last_model_checkpoint}')
