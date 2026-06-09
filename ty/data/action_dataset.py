import random
import torch.utils.data as Data
import os
import numpy as np
import torch
import torch.nn.functional as F
import re
from PIL import Image, ImageDraw, ImageFilter
from torchvision.transforms import transforms
from ty.util.util import update_progress, resize_bbox, patches_union_pixel_bbox_from_bbox
from ty.PAA_data_process.utils import jitter_bbox, squarify



class ActionDataset(Data.Dataset):
    def __init__(self, args, mode, raw_data, configs, dataset):
        super(ActionDataset, self).__init__()

        self.mode = mode
        self.args = args
        self.dataset = dataset
        self.raw_data = raw_data
        self.configs = configs

        self.model_opts = configs['model_opts']
        data, neg_count, pos_count = self.get_data_sequence(mode, raw_data, self.model_opts)
        self.ped_samples = data['raw_image']
        self.crop_ped_image = data['crop_ped_image']
        self.new_bbox = data['new_bbox']
        # self.ped_id = data['ped_id']
        self.labels = data['crossing']
        self.use_all_bbox = args.use_all_bbox
        self.crossing_point_bbox = data['crossing_point_bbox']
        self.past_bbox = data['past_bbox']
        self.box_org = data['box_org']

        print(f"number of {self.mode} sample pedestrians: {self.ped_samples.shape[0]}\n")

    def get_data_sequence(self, data_type, data_raw, opts):
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        augmentation = opts['data_augmentation'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        # time_to_event = 55
        normalize = opts['normalize_boxes']
        # cross_point_bbox
        cross_point_bbox_dict = [{sub_a[-1][0]: sub_b[-1]} for sub_a, sub_b in zip(data_raw['pid'], data_raw['bbox'])]

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')
        if augmentation:
            self.data_augmentation_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
        else:
            # if data_type == 'train':
            #     overlap = opts['overlap'] # if data_type == 'train' else 0.0
            # else:
            #     overlap = 0.0
            overlap = opts['overlap']  # if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs

            for seq in data_raw['bbox']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                range(start_idx, end_idx + 1, olap_res)])

        cross_point_bbox_list = []
        for sub_a in d['ped_id']:
            key = sub_a[0][0]
            for dict in cross_point_bbox_dict:
                if key in dict:
                    cross_point_bbox_list.append(dict[key])
                    break
        W, H = 1920, 1080
        cross_point_bbox_list = [[x1 / W, y1 / H, x2 / W, y2 / H] for x1, y1, x2, y2 in cross_point_bbox_list]
        d['crossing_point_bbox'] = cross_point_bbox_list
        past_bbox_list = [[[x1 / W, y1 / H, x2 / W, y2 / H] for x1, y1, x2, y2 in b] for b in d['box']]
        d['past_bbox'] = past_bbox_list
        if normalize:
            for k in d.keys():
                if k != 'tte':
                    if k != 'box' and k != 'center':
                        for i in range(len(d[k])):
                            d[k][i] = d[k][i][1:]
                    else:
                        for i in range(len(d[k])):
                            d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                d[k] = np.array(d[k])
        else:
            for k in d.keys():
                d[k] = np.array(d[k])

        # process_image
        raw_images, crop_ped_image, new_bbox = self.preprocess(d)
        d['raw_image'] = raw_images
        d['crop_ped_image'] = crop_ped_image
        d['new_bbox'] = new_bbox

        d['crossing'] = np.array(d['crossing'])[:, 0, :]
        pos_count = np.count_nonzero(d['crossing'])
        neg_count = len(d['crossing']) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def preprocess(self, data):
        img_sequences = data['image']
        bbox = data['box']
        ped_id = data['ped_id']

        raw_sequences = []
        crop_ped_sequences = []
        new_bbox_sequences = []
        i = -1
        for seq, pid in zip(img_sequences, ped_id):
            i += 1
            update_progress(i / len(img_sequences))
            raw_seq = []
            crop_ped_seq = []
            new_bbox_seq = []
            j = -1
            for imp, b, p in zip(seq, bbox[i], pid):
                b = [int(x) for x in b]
                j += 1
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]

                # raw_resize_img_save_folder = os.path.join('/data1/ty/code/PAA/data/VRPE_4_C2F_patch_correspondence/bgb0.5_pan224112_ped112112_bboxenlarge1.5/pan', self.dataset, set_id, vid_id)
                raw_resize_img_save_folder = os.path.join(
                    '/data1/ty/code/PAA/data/VRPE_4_C2F_patch_correspondence',
                    f"bgb{self.args.bg_downsample_ratio}_pan{self.args.pan_resolution_w}{self.args.pan_resolution_h}"
                    f"_panps{self.args.pan_patch_size}_pedps{self.args.ped_patch_size}"
                    f"_ped{self.args.ped_resolution_w}{self.args.ped_resolution_h}_bboxenlarge{self.args.crop_resize_ratio}",
                    'pan',
                    self.dataset,
                    set_id,
                    vid_id
                )
                os.makedirs(raw_resize_img_save_folder, exist_ok=True)
                raw_resize_img_save_path = os.path.join(raw_resize_img_save_folder, img_name + '_' + p[0] + '.png')

                # crop_ped_resize_img_save_folder = os.path.join('/data1/ty/code/PAA/data/VRPE_4_C2F_patch_correspondence/nobgb_pan224112_ped112112_bboxenlarge1.5/ped',self.dataset, set_id, vid_id)
                crop_ped_resize_img_save_folder = os.path.join(
                    '/data1/ty/code/PAA/data/VRPE_4_C2F_patch_correspondence',
                    f"bgb{self.args.bg_downsample_ratio}_pan{self.args.pan_resolution_w}{self.args.pan_resolution_h}"
                    f"_panps{self.args.pan_patch_size}_pedps{self.args.ped_patch_size}"
                    f"_ped{self.args.ped_resolution_w}{self.args.ped_resolution_h}_bboxenlarge{self.args.crop_resize_ratio}",
                    'ped',
                    self.dataset,
                    set_id,
                    vid_id
                )
                os.makedirs(crop_ped_resize_img_save_folder, exist_ok=True)
                crop_ped_resize_save_path = os.path.join(crop_ped_resize_img_save_folder, img_name + '_' + p[0] + '.png')

                bboxed_img_save_folder = os.path.join('/data1/ty/code/PAA/data/bbox_image', self.dataset, set_id, vid_id)
                os.makedirs(crop_ped_resize_img_save_folder, exist_ok=True)
                bboxed_img_save_path = os.path.join(bboxed_img_save_folder, img_name + '_' + p[0] + '.png')

                # Expand the bbox of the target pedestrian in the first frame to make it the bbox of the target pedestrian detected in subsequent frames
                if j==0:
                    # firstframe_pan_bboxed_resize = os.path.join('/data1/ty/code/PAA/data/VRPE_4_C2F_patch_correspondence/bgb0.5_pan224112_ped112112_bboxenlarge1.5/firstframe_pan', self.dataset, set_id, vid_id)
                    firstframe_pan_bboxed_resize = os.path.join(
                        '/data1/ty/code/PAA/data/VRPE_4_C2F_patch_correspondence',
                        f"bgb{self.args.bg_downsample_ratio}_pan{self.args.pan_resolution_w}{self.args.pan_resolution_h}"
                        f"_panps{self.args.pan_patch_size}_pedps{self.args.ped_patch_size}"
                        f"_ped{self.args.ped_resolution_w}{self.args.ped_resolution_h}_bboxenlarge{self.args.crop_resize_ratio}",
                        'firstframe_pan',
                        self.dataset,
                        set_id,
                        vid_id
                    )
                    os.makedirs(firstframe_pan_bboxed_resize, exist_ok=True)
                    firstframe_pan_bboxed_resize = os.path.join(firstframe_pan_bboxed_resize,img_name + '_' + p[0] + '.png')

                    # firstframe_ped_bboxed_resize = os.path.join('/data1/ty/code/PAA/data/VRPE_4_C2F_patch_correspondence/nobgb_pan224112_ped112112_bboxenlarge1.5/firseframe_ped', self.dataset, set_id, vid_id)
                    firstframe_ped_bboxed_resize = os.path.join(
                        '/data1/ty/code/PAA/data/VRPE_4_C2F_patch_correspondence',
                        f"bgb{self.args.bg_downsample_ratio}_pan{self.args.pan_resolution_w}{self.args.pan_resolution_h}"
                        f"_panps{self.args.pan_patch_size}_pedps{self.args.ped_patch_size}"
                        f"_ped{self.args.ped_resolution_w}{self.args.ped_resolution_h}_bboxenlarge{self.args.crop_resize_ratio}",
                        'firstframe_ped',
                        self.dataset,
                        set_id,
                        vid_id
                    )
                    os.makedirs(firstframe_ped_bboxed_resize, exist_ok=True)
                    firstframe_ped_bboxed_resize = os.path.join(firstframe_ped_bboxed_resize,img_name + '_' + p[0] + '.png')

                    first_bbox_enlarge = jitter_bbox(imp, [b], 'enlarge', self.args.crop_resize_ratio)[0]
                    first_bbox_enlarge = squarify(first_bbox_enlarge, 1, 1920)
                    first_bbox_enlarge = list(map(int, first_bbox_enlarge[0:4]))
                    raw_resize_img_save_path = firstframe_pan_bboxed_resize
                    crop_ped_resize_save_path = firstframe_ped_bboxed_resize

                # Check whether the file exists
                if not os.path.exists(raw_resize_img_save_path) or not os.path.exists(crop_ped_resize_save_path) :
                    if j == 0 :
                        if not os.path.exists(bboxed_img_save_path):
                            bboxed_img_save_path = self.draw_and_save_bbox(imp, b, img_name, p, set_id, vid_id)
                        imp = bboxed_img_save_path

                    if j != 0 and 'flip' in imp:
                        imp = imp.replace('_flip', '')
                        raw_imp = Image.open(imp)
                        # image.show()
                        raw_imp = raw_imp.transpose(Image.FLIP_LEFT_RIGHT)
                    else:
                        raw_imp = Image.open(imp)   # (1080, 1920, 3)

                    # x1, y1, x2, y2 = first_bbox_enlarge
                    coarse_b = resize_bbox(first_bbox_enlarge, orig_w=1920, orig_h=1080, target_w=self.args.pan_resolution_w, target_h=self.args.pan_resolution_h)


                    # person = raw_imp.crop((x1, y1, x2, y2))
                    # try:
                    #     if j != 0 and 'flip' in imp:
                    #         imp = imp.replace('_flip', '')
                    #         raw_imp = Image.open(imp)
                    #         raw_imp = raw_imp.transpose(Image.FLIP_LEFT_RIGHT)
                    #     else:
                    #         raw_imp = Image.open(imp)  # (1080, 1920, 3)
                    #
                    #     x1, y1, x2, y2 = first_bbox_enlarge
                    #     person = raw_imp.crop((x1, y1, x2, y2))
                    #
                    # except Exception as e:
                    #     print(f"\n[❌ Error loading image] {imp}")
                    #     print(f"Error type: {type(e).__name__}, message: {e}\n")
                    #     raise e  # 让程序中断并保留完整堆栈
                    patches_union_pixel_bbox = patches_union_pixel_bbox_from_bbox(coarse_b, W=self.args.pan_resolution_w, H=self.args.pan_resolution_h,P_size=self.args.pan_patch_size)
                    coarse_b_Original_img_b = resize_bbox(patches_union_pixel_bbox, orig_w=self.args.pan_resolution_w, orig_h=self.args.pan_resolution_h,target_w=1920, target_h=1080)
                    x1, y1, x2, y2 = coarse_b_Original_img_b
                    imp_bg_blurry = self.process_frame(raw_imp, coarse_b_Original_img_b, bg_downsample_ratio=self.args.bg_downsample_ratio)
                    raw_resize_imp = imp_bg_blurry.resize((self.args.pan_resolution_w, self.args.pan_resolution_h), Image.BICUBIC)
                    person = raw_imp.crop((x1, y1, x2, y2))
                    person_resize_imp = person.resize((self.args.ped_resolution_w, self.args.ped_resolution_h), Image.BICUBIC)

                    raw_resize_imp.save(raw_resize_img_save_path)
                    person_resize_imp.save(crop_ped_resize_save_path)

                new_b = resize_bbox(first_bbox_enlarge, orig_w=1920, orig_h=1080, target_w=self.args.pan_resolution_w, target_h=self.args.pan_resolution_h)
                raw_seq.append(raw_resize_img_save_path)
                crop_ped_seq.append(crop_ped_resize_save_path)
                new_bbox_seq.append(new_b)

            raw_sequences.append(raw_seq)
            crop_ped_sequences.append(crop_ped_seq)
            new_bbox_sequences.append(new_bbox_seq)
        raw_sequences = np.array(raw_sequences)
        crop_ped_sequences = np.array(crop_ped_sequences)
        new_bbox_sequences = np.array(new_bbox_sequences)
        return raw_sequences, crop_ped_sequences, new_bbox_sequences

    def process_frame(self, img, bbox,
                      bg_downsample_ratio=0.1,
                      feather=10):

        # --- 参数与图像尺寸 ---
        w, h = img.size
        x1, y1, x2, y2 = bbox

        # --- 背景下采样再放大（相当于模糊效果） ---
        bg_small = img.resize(
            (max(1, int(w * bg_downsample_ratio)), max(1, int(h * bg_downsample_ratio))),
            resample=Image.BILINEAR
        )
        bg_blur = bg_small.resize((w, h), resample=Image.BILINEAR)

        # --- 提取行人区域 ---
        person = img.crop((x1, y1, x2, y2))

        # --- 将行人粘回到背景上 ---
        fused = bg_blur.copy()
        fused.paste(person, (x1, y1, x2, y2))

        # --- 创建羽化 mask ---
        mask = Image.new("L", (w, h), 0)  # 灰度 mask
        mask_draw = Image.new("L", (x2 - x1, y2 - y1), 255)
        mask.paste(mask_draw, (x1, y1, x2, y2))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

        # --- 融合图像 (PIL支持alpha混合) ---
        fused = Image.composite(fused, bg_blur, mask)
        return fused


    def draw_and_save_bbox(self, imp, bbox, img_name, ped_id, set_id, vid_id):
        bbox_img_path = os.path.join(self.args.path_bbox_image, self.model_opts['dataset'], set_id, vid_id)
        os.makedirs(bbox_img_path, exist_ok=True)
        bbox_imp = os.path.join(bbox_img_path, img_name + '_' + ped_id[0] + '.png')

        if not os.path.exists(bbox_imp):
            image = Image.open(imp)
            draw = ImageDraw.Draw(image)
            draw.rectangle(bbox, outline="red", width=3)
            image.save(bbox_imp)

        return bbox_imp

    def __getitem__(self, idx):

        images = self.ped_samples[idx]
        pan_images = images.tolist()
        crop_ped_images = self.crop_ped_image[idx]
        crop_ped_images = crop_ped_images.tolist()
        new_bbox = self.new_bbox[idx]
        new_bbox = new_bbox.tolist()
        label = self.labels[idx].item()
        crossing_point_bbox = self.crossing_point_bbox[idx]
        crossing_point_bbox = crossing_point_bbox.tolist()
        past_bbox = self.past_bbox[idx]
        past_bbox = past_bbox.tolist()
        # raw_bbox = self.box_org[idx][0]
        # raw_bbox = raw_bbox.tolist()
        # raw_bbox = [int(x) for x in raw_bbox]

        # query = 'Given 16 sequential front-camera images from an autonomous vehicle, ' \
        #         'identify the one pedestrian marked by a red box in the first frame and track that pedestrian across all frames (consider traffic lights, vehicles, etc.). ' \
        #         'Determine if this pedestrian will cross the street.  ' \
        #         'Output exactly "crossing" or "not crossing" with no additional text.'
        # query = 'Given 16 sequential front-camera images from an autonomous vehicle, ' \
        #         'where the target pedestrian is marked by a red box in the first frame, track the target pedestrian across all frames. ' \
        #         '(bbox coordinates: [{}, {}, {}, {}]),' \
        #         'Determine if this pedestrian will cross the street (consider movement patterns, behavioral cues, and environmental context). ' \
        #         'Output exactly "crossing" or "not crossing" with no additional text.'
        # query = query.format(*raw_bbox)

        if self.args.prompt_length == 'short':
            query = 'Track the pedestrian marked in the first frame across 16 images' \
                    'and decide if he/she will cross the street.' \
                    'Reply only "crossing" or "not crossing".'
        elif self.args.prompt_length == 'medium':
            query = 'Given 16 sequential front-camera images from an autonomous vehicle, ' \
                    'where the target pedestrian is marked by a red box in the first frame, track the target pedestrian across all frames. ' \
                    'Determine if this pedestrian will cross the street (consider movement patterns, behavioral cues, and environmental context). ' \
                    'Output exactly "crossing" or "not crossing" with no additional text.'
        elif self.args.prompt_length == 'long':
            query = "Predict if the target pedestrian will cross the street based on the provided 16-frame sequence from the ego vehicle's front camera." \
                    "Identify the single target pedestrian (marked with a red box in the first frame) and track this individual across all 16 frames." \
                    "Examine the target pedestrian's evolving trajectory, behavioral cues (e.g., changes in walking pace, pauses, gaze direction), and appearance (posture, gestures)." \
                    "Assess environmental context including traffic light status, crosswalk proximity, vehicle movements, and nearby obstacles." \
                    "Synthesize these spatio-temporal dynamics to forecast crossing behavior." \
                    "Output exactly 'crossing' or 'not crossing' with no additional text."

        if label == 1:
            response = 'crossing'
        elif label == 0:
            response = 'not crossing'

        return {"query": query, "response": response, "images": pan_images, "crop_ped_images": crop_ped_images, "new_bbox": new_bbox, "crossing_point_bbox" : crossing_point_bbox, "past_bbox": past_bbox}

    def __len__(self):
        return self.ped_samples.shape[0]

    def shuffle_list(self, list):
        random.shuffle(list)