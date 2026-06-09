import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream, inference_class
)
from swift.utils import seed_everything
from swift.tuners import Swift
import torch
from ty.data.action_dataset import ActionDataset
from tqdm import tqdm
from ty.data.action_dataset import ActionDataset
import yaml
from ty.PAA_data_process.jaad_data import JAAD
from ty.PAA_data_process.pie_data import PIE
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import OrderedDict
from openpyxl import load_workbook
from safetensors.torch import load_file
import time
from swift.llm.utils.utils import LazyLLMDataset
import os
import cv2
import torch
import numpy as np


def infer(args):
    seed_everything(42)
    model_type = ModelType.qwen2_vl_7b_instruct
    model_id_or_path = args.model_id_or_path
    # model_id_or_path = '/home/ty/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct'
    # model_id_or_path = '/home/ty/.cache/modelscope/hub/qwen/Qwen2-VL-2B-Instruct'
    template_type = get_default_template_type(model_type)
    print(f'template_type: {template_type}')

    ckpt_dir = args.ckpt_dir
    # pos_weight
    if args.dataset == 'pie':
        pos_weight = 3576.0 / 1194.0  # pie
    if args.dataset == 'jaad_beh':
        pos_weight = 374.0 / 1760.0  # jaad_beh
    if args.dataset == 'jaad_all':
        pos_weight = 6853.0 / 1760.0  # jaad_all
    is_predict = True
    is_obs_bbox = False
    is_pre_bbox = False
    is_classifier =True

    model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                           model_kwargs={'device_map': 'auto'},
                                           model_id_or_path=model_id_or_path, is_predict=is_predict,
                                           pos_weight=pos_weight, pan_resolution_h=args.pan_resolution_h,
                                           pan_resolution_w=args.pan_resolution_w, pan_patch_size=args.pan_patch_size,
                                           ped_resolution_h=args.ped_resolution_h,
                                           ped_resolution_w=args.ped_resolution_w, ped_patch_size=args.ped_patch_size,
                                           lossweight_class=args.lossweight_class,lossweight_pre=args.lossweight_pre,
                                           lossweight_obs=args.lossweight_obs,
                                           is_obs_bbox = is_obs_bbox,
                                           is_pre_bbox = is_pre_bbox,
                                           is_classifier=is_classifier
                                           )

    model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)   # TY

    extra_ck_path = os.path.join(ckpt_dir, 'extra_layer.safetensors')
    state_dict = load_file(extra_ck_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    model.generation_config.max_new_tokens = 500
    template = get_template(template_type, tokenizer)

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
    # tem
    # raw_data_small_data = {key: value[89:94] for key, value in raw_data_test.items()}
    # raw_data_small_data["image_dimension"] = (1920, 1080)
    test_dataset = ActionDataset(args, 'test', raw_data_test, configs, dataset = configs['model_opts']['dataset'])
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=False,
        shuffle=False
    )
    all_probs = []
    all_labels = []
    with torch.no_grad():
        total = 0
        correct = 0
        for data in tqdm(data_loader):
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
            past_bbox_label = data["past_bbox"]
            past_bbox_label = torch.tensor(past_bbox_label)
            text_label = data["response"][0]
            future_bbox_label = data["crossing_point_bbox"]
            future_bbox_label = torch.tensor(future_bbox_label)

            # print(imp_list)
            # print("--------------------------------------------------")

            # response, history = inference(
            #     model,
            #     template,
            #     query,
            #     images=imp_list,
            #     system=args.system,
            #     crop_ped_images=crop_imp_list,
            #     new_bbox=new_bbox_list
            # )
            #
            # print(f"'response': {response}, 'label': {text_label}", file=f)
            # print("--------------------------------------------------", file=f)

            class_logits = inference_class(
                model,
                template,
                query,
                images=imp_list,
                system=args.system,
                crop_ped_images=crop_imp_list,
                new_bbox=new_bbox_list
            )

            probs = torch.sigmoid(class_logits.squeeze(-1))
            eval_class = (probs >= 0.5).long()
            eval_class = eval_class.cpu()

            class_label = []
            if text_label == "crossing":
                class_label.append(1)
            elif text_label == "not crossing":
                class_label.append(0)

            pred_label = "crossing" if eval_class.item ==1 else "not crossing"
            true_label = text_label

            print(f"prediction : {pred_label}", f", true: {true_label}")


            all_probs.append(probs.detach().cpu().numpy())
            class_label = torch.tensor(class_label)
            all_labels.append(class_label.cpu().numpy())

            total += 1
            eval_class = eval_class.to(torch.int64)
            correct += (eval_class == class_label).sum().item()

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        all_preds = (all_probs >= 0.5).astype(int)

        acc = correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds)

        # average = [acc, f1, precision, recall]
        # average = np.mean(average)

        metrics = {
            'acc': "{:.2f}".format(acc),
            'f1': "{:.2f}".format(f1),
            'precision': "{:.2f}".format(precision),
            'recall': "{:.2f}".format(recall),
        }
        print(metrics)


if __name__ == '__main__':
    parser = ArgumentParser(description="Train-Test program for PAAMLM")
    parser.add_argument('--path_dataset_config', type=str, default="/home/ty/code/ty/code/PAA/pekinese/PAALLM_V4_1/ty/configs")
    parser.add_argument("--dataset", type=str, default='jaad_beh', help="Dataset to use. Choices: 'pie', 'jaad_all', 'jaad_beh'")
    parser.add_argument("--use_all_bbox", type=bool, default=False)
    parser.add_argument('--path_bbox_image', type=str, default="/data1/ty/code/PAA/data/bbox_image")
    # parser.add_argument('--ckpt_dir', type=str, default="/data1/ty/code/PAA/pekinese/PAALLM_V4_3_ablation_2/ty/output/jaad_all/short_bgb0.1_1coarseattnmapfine_fp16_pan44822414_ped1121127_b1.5/checkpoint-3766")  # jaad_all
    parser.add_argument('--ckpt_dir', type=str, default="/data1/ty/code/PAA/pekinese/PAALLM_V4_3_ablation_1/ty/output/ablation/mlm/7Blora_short_lossweight10.50.50.5_bgb0.5_pan44822414_ped1121127_b1.5/checkpoint-266")  # jaad_beh
    parser.add_argument('--metric_output_file', type=str, default="")
    parser.add_argument("--crop_resize_ratio", type=float, default=1.5)
    parser.add_argument("--bg_downsample_ratio", type=float, default=0.1)
    parser.add_argument("--pan_resolution_h", type=int, default=224)
    parser.add_argument("--pan_resolution_w", type=int, default=448)
    parser.add_argument("--pan_patch_size", type=int, default=14)
    parser.add_argument("--ped_resolution_h", type=int, default=112)
    parser.add_argument("--ped_resolution_w", type=int, default=112)
    parser.add_argument("--ped_patch_size", type=int, default=7)
    parser.add_argument("--lossweight_class", type=float, default=0.5)
    parser.add_argument("--lossweight_pre", type=float, default=0.5)
    parser.add_argument("--lossweight_obs", type=float, default=0.5)
    parser.add_argument("--prompt_length", type=str, default='short')
    parser.add_argument("--model_id_or_path", type=str, default='/home/ty/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument("--system", type=str, default='You are an autonomous driving system.')
    args = parser.parse_args()
    infer(args)

