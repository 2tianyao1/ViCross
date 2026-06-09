import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pandas as pd
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
from swift.tuners import Swift
import torch
from ty.data.action_dataset import ActionDataset
from tqdm import tqdm
from ty.data.action_dataset import ActionDataset
import yaml
from ty.PAA_data_process.jaad_data import JAAD
# from ty.data.jaad_data import JAAD
# from ty.PAA_data_process.pie_data import PIE
from ty.data.pie_data import PIE
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import OrderedDict
from openpyxl import load_workbook
from safetensors.torch import load_file
import time
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count
def infer(args):
    seed_everything(args.seed)
    model_type = ModelType.qwen2_vl_7b_instruct
    model_id_or_path = args.model_id_or_path
    # model_id_or_path = '/home/ty/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct'
    # model_id_or_path = '/home/ty/.cache/modelscope/hub/qwen/Qwen2-VL-2B-Instruct'
    template_type = get_default_template_type(model_type)
    print(f'template_type: {template_type}')

    # model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
    #                                        model_kwargs={'device_map': 'auto'},load_model=True, model_id_or_path="/data1/ty/Qwen/data_1/Qwen2-VL-7B-Instruct",)
    ckpt_dir = args.ckpt_dir
    # pos_weight
    if args.dataset == 'pie':
        pos_weight = 3576.0 / 1194.0  # pie
    if args.dataset == 'jaad_beh':
        pos_weight = 374.0 / 1760.0  # jaad_beh
    if args.dataset == 'jaad_all':
        pos_weight = 6853.0 / 1760.0  # jaad_all
    is_predict = True
    # model, tokenizer = get_model_tokenizer(model_type, torch.float16,
    #                                        model_kwargs={'device_map': 'auto'},
    #                                        model_id_or_path=model_id_or_path, is_predict = is_predict, pos_weight = pos_weight)
    model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                           model_kwargs={'device_map': 'auto'},
                                           model_id_or_path=model_id_or_path, is_predict=is_predict,
                                           pos_weight=pos_weight, pan_resolution_h=args.pan_resolution_h,
                                           pan_resolution_w=args.pan_resolution_w, pan_patch_size=args.pan_patch_size,
                                           ped_resolution_h=args.ped_resolution_h,
                                           ped_resolution_w=args.ped_resolution_w, ped_patch_size=args.ped_patch_size,
                                           lossweight_class=args.lossweight_class,lossweight_pre=args.lossweight_pre,
                                           lossweight_obs=args.lossweight_obs
                                           )

    model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)   # TY

    extra_ck_path = os.path.join(ckpt_dir, 'extra_layer.safetensors')
    state_dict = load_file(extra_ck_path)
    model.load_state_dict(state_dict, strict=False)

    model.generation_config.max_new_tokens = 500
    template = get_template(template_type, tokenizer)
    # seed_everything(42)
    dataset = args.dataset
    dataset_config = os.path.join(args.path_dataset_config, f"config_{args.dataset}.yaml")
    with open(dataset_config, 'r') as f:
        configs = yaml.safe_load(f)

    # Calculate min track size
    tte = configs['model_opts']['time_to_event'] if isinstance(configs['model_opts']['time_to_event'], int) else \
        configs['model_opts']['time_to_event'][1]
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + tte

    if configs['model_opts']['dataset'] == 'pie':
        imdb = PIE(data_path="/data1/ty/code/PAA/data/PIE")
    elif configs['model_opts']['dataset'] == 'jaad':
        imdb = JAAD(data_path="/data1/ty/code/PAA/data/JAAD")

    raw_data_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])
    test_dataset = ActionDataset(args, 'test', raw_data_test, configs, dataset = configs['model_opts']['dataset'])

    data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                drop_last=False,
                shuffle=False
            )
    y_true = []
    y_pred = []
    response_abnormal = 0
    res_des = dict()
    i=0
    for data in tqdm(data_loader):
        # i+=1
        # if i==60:
        #     a=1
        images = data["images"]
        crop_ped_images = data["crop_ped_images"]
        new_bbox = data["new_bbox"]
        imp_list = []
        crop_imp_list = []
        for tu_imp in images:
            imp = tu_imp[0]
            imp_list.append(imp)
        for tu_crop_imp in crop_ped_images:
            crop_imp = tu_crop_imp[0]
            crop_imp_list.append(crop_imp)
        new_bbox_list = []
        for bbox_per_image in new_bbox:
            bbox = []
            for coordinate in bbox_per_image:
                bbox.append(coordinate[0])
            new_bbox_list.append(bbox)
        query = data["query"][0]
        label = data["response"][0]
        # start = time.time()
        response, history = inference(model, template, query, images = imp_list, system = args.system, crop_ped_images = crop_imp_list, new_bbox = new_bbox_list)
        # end = time.time()
        # print(f"运行时间: {end - start:.6f} s")

        # print(imp_list)
        print(f"'response': {response}, 'label':{label}")
        if response != 'crossing' and response != 'not crossing':
            response_abnormal +=1

        label = 1 if label == "crossing" else 0
        response = 1 if response == "crossing" else 0
        y_true.append(label)
        y_pred.append(response)

    npy_pred = np.array(y_pred)
    npy_true = np.array(y_true)

    df = pd.DataFrame(npy_pred)
    df.columns = [None] * df.shape[1]  # 将列名设置为 None
    save_path = "/home/ty/code/ty/code/PAA/pekinese/PAALLM_V4_3_ablation_1/ty/output/vis_for_scenior/APSR_pie_for_scenario_pre.xlsx"
    df.to_excel(save_path, index=False, header = False)
    print(f"Saved prediction results to {save_path}")


    ck_num = next(
        (int(p.split('-')[-1]) for p in args.ckpt_dir.split('/')
         if 'checkpoint-' in p),
        None
    )
    res_des['ck'] = ck_num
    accuracy = accuracy_score(y_true, y_pred)
    acc_formatted = "{:.4f}".format(accuracy)
    res_des['acc'] = acc_formatted

    f1 = f1_score(y_true, y_pred)
    f1_formatted = "{:.4f}".format(f1)
    res_des['f1'] = f1_formatted

    precision = precision_score(y_true, y_pred)
    pre_formatted = "{:.4f}".format(precision)
    res_des['pre'] = pre_formatted

    recall = recall_score(y_true, y_pred)
    rec_formatted = "{:.4f}".format(recall)
    res_des['rec'] = rec_formatted
    res_des['res_abnormal'] = response_abnormal

    print(f"Dataset: {dataset}")
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"response_abnormal: {response_abnormal}")

    res_des = OrderedDict(res_des)
    data = [item for pair in res_des.items() for item in pair]

    metric_output_file = args.metric_output_file
    sheet_name = 'Sheet1'

    workbook = load_workbook(metric_output_file)
    sheet = workbook[sheet_name]
    sheet.append(data)

    workbook.save(metric_output_file)

if __name__ == '__main__':
    parser = ArgumentParser(description="Train-Test program for PAAMLM")
    parser.add_argument('--path_dataset_config', type=str, default="/home/ty/code/ty/code/PAA/pekinese/PAALLM_V4_1/ty/configs")
    parser.add_argument("--dataset", type=str, default='jaad_beh', help="Dataset to use. Choices: 'pie', 'jaad_all', 'jaad_beh'")
    parser.add_argument("--use_all_bbox", type=bool, default=False)
    parser.add_argument('--path_bbox_image', type=str, default="/data1/ty/code/PAA/data/bbox_image")
    parser.add_argument('--ckpt_dir', type=str, default="/data1/ty/code/PAA/pekinese/PAALLM_V4_3_ablation_1/ty/output/ablation/mlm/7Blora_short_lossweight10.50.50.5_bgb0.5_pan44822414_ped1121127_b1.5/checkpoint-266")
    parser.add_argument('--metric_output_file', type=str, default="")
    parser.add_argument("--crop_resize_ratio", type=float, default=1.5)
    parser.add_argument("--bg_downsample_ratio", type=float, default=0.5)
    parser.add_argument("--pan_resolution_h", type=int, default=224)
    parser.add_argument("--pan_resolution_w", type=int, default=448)
    parser.add_argument("--pan_patch_size", type=int, default=14)
    parser.add_argument("--ped_resolution_h", type=int, default=112)
    parser.add_argument("--ped_resolution_w", type=int, default=112)
    parser.add_argument("--ped_patch_size", type=int, default=7)
    parser.add_argument("--lossweight_class", type=float, default=1)
    parser.add_argument("--lossweight_pre", type=float, default=1)
    parser.add_argument("--lossweight_obs", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_length", type=str, default='short')
    parser.add_argument("--model_id_or_path", type=str, default='/home/ty/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument("--system", type=str, default='You are an autonomous driving system.')
    args = parser.parse_args()
    infer(args)

