from PIL import Image


def res_image(img, target=224):
    ratio = min(target/img.size[0], target/img.size[1])
    new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
    return img.resize(new_size, Image.BICUBIC)


def patches_union_pixel_bbox_from_bbox(bbox, P_size=14, img_size=(224, 112)):
    """
    bbox: [x0, y0, x1, y1]  （按你代码使用的语义：x1,y1 可被视作 exclusive 或 inclusive 都可，代码用 (x1-1)//P）
    P_size: patch size (int)
    img_size: (W, H)
    Returns: [px_x0, px_y0, px_x1, px_y1]  (inclusive coordinates, clamped to image)
    """
    W, H = img_size
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