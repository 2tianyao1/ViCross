import sys
from PIL import Image

def update_progress(progress):
    """
    Shows the progress
    Args:
        progress: Progress thus far
    """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)

    block = int(round(barLength * progress))
    text = "\r[{}] {:0.2f}% {}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def resize_bbox(bbox, orig_w=1920, orig_h=1080, target_w=224, target_h=112):
    """
    将原图(1920x1080)中的bbox映射到强制resize后的图像(224x112)

    Args:
        bbox: [x1, y1, x2, y2] 原图坐标
        orig_w, orig_h: 原图宽高
        target_w, target_h: 缩放后的目标宽高

    Returns:
        [new_x1, new_y1, new_x2, new_y2] 缩放后的坐标
    """
    x1, y1, x2, y2 = bbox

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    new_x1 = int(x1 * scale_x)
    new_y1 = int(y1 * scale_y)
    new_x2 = int(x2 * scale_x)
    new_y2 = int(y2 * scale_y)

    return [new_x1, new_y1, new_x2, new_y2]

def res_image(img, target=224):
    ratio = min(target/img.size[0], target/img.size[1])
    new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
    return img.resize(new_size, Image.BICUBIC)

def bbox_sanity_check(img_size, bbox):
    """
    Checks whether  bounding boxes are within image boundaries.
    If this is not the case, modifications are applied.
    Args:
        img_size: The size of the image
        bbox: The bounding box coordinates
    Return:
        The modified/original bbox
    """
    img_width, img_heigth = img_size
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_width:
        bbox[2] = img_width - 1
    if bbox[3] >= img_heigth:
        bbox[3] = img_heigth - 1
    return bbox

def patches_union_pixel_bbox_from_bbox(bbox, W, H, P_size=14,):
    """
    bbox: [x0, y0, x1, y1]  （按你代码使用的语义：x1,y1 可被视作 exclusive 或 inclusive 都可，代码用 (x1-1)//P）
    P_size: patch size (int)
    img_size: (W, H)
    Returns: [px_x0, px_y0, px_x1, px_y1]  (inclusive coordinates, clamped to image)
    """
    x0, y0, x1, y1 = bbox

    # compute patch indices exactly like your code (0-based)
    c0 = int(x0 // P_size)
    c1 = int((x1 - 1) // P_size)
    r0 = int(y0 // P_size)
    r1 = int((y1 - 1) // P_size)

    # clamp patch indices to valid grid
    max_c = W // P_size - 1
    max_r = H // P_size - 1
    if c0 < 0: c0 = 0
    if r0 < 0: r0 = 0
    if c1 > max_c: c1 = max_c
    if r1 > max_r: r1 = max_r

    # pixel bbox covered by the patches (inclusive)
    px_x0 = c0 * P_size
    px_y0 = r0 * P_size
    px_x1 = (c1 + 1) * P_size - 1
    px_y1 = (r1 + 1) * P_size - 1

    # clamp to image pixel extents
    px_x0 = max(0, min(px_x0, W - 1))
    px_y0 = max(0, min(px_y0, H - 1))
    px_x1 = max(0, min(px_x1, W - 1))
    px_y1 = max(0, min(px_y1, H - 1))

    return [px_x0, px_y0, px_x1, px_y1]