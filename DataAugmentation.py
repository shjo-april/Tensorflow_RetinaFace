# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import random

import numpy as np

# DataAugmentation (threshold = 0 ~ 1, 0 ~ 100%)

# use
FLIP_HORIZONTAL = 0.5
GAUSSIAN_NOISE = 0.1
CROP = 0.20

# not use
FLIP_VERTICAL = 0.2
SCALE = 0.1
BRIGHTNESS = 0.25
HUE = 0.50
SATURATION = 0.25
GRAY = 0.1
SHIFT = 0.25

def random_horizontal_flip(image, gt_bboxes = None, threshold = FLIP_HORIZONTAL):
    if random.random() <= threshold:
        image = cv2.flip(image, 1).copy()
        
        if gt_bboxes is not None:
            h, w, c = image.shape
            gt_bboxes[:, 0], gt_bboxes[:, 2] = w - gt_bboxes[:, 2], w - gt_bboxes[:, 0]

    if gt_bboxes is not None:
        return image, gt_bboxes.astype(np.float32)
    else:
        return image

def random_vertical_flip(image, gt_bboxes = None, threshold = FLIP_VERTICAL):
    if random.random() <= threshold:
        image = cv2.flip(image, 0).copy()

        if gt_bboxes is not None:
            h, w, c = image.shape
            gt_bboxes[:, 1], gt_bboxes[:, 3] = h - gt_bboxes[:, 3], h - gt_bboxes[:, 1]

    if gt_bboxes is not None:
        return image, gt_bboxes.astype(np.float32)
    else:
        return image

def random_scale(image, gt_bboxes = None, threshold = SCALE):
    if random.random() <= threshold:
        w_scale = random.uniform(0.75, 1.25)
        h_scale = random.uniform(0.75, 1.25)

        image = cv2.resize(image, None, fx = w_scale, fy = h_scale, interpolation = cv2.INTER_CUBIC)

        if gt_bboxes is not None:
            gt_bboxes = gt_bboxes * [w_scale, h_scale, w_scale, h_scale]

    if gt_bboxes is not None:
        return image, gt_bboxes.astype(np.float32)
    else:
        return image

def random_brightness(image, threshold = BRIGHTNESS):
    if random.random() <= threshold:
        scale = random.uniform(0.5, 1.5)
        image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    return image

def random_hue(image, h_range = 36, threshold = HUE):
    if random.random() <= threshold:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        h_scale = random.uniform(-h_range, h_range)
        h = np.clip(h + h_scale, 0, 360).astype(np.uint8)

        hsv_image = np.stack([h, s, v]).transpose((1, 2, 0))
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return image

def random_saturation(image, threshold = SATURATION):
    if random.random() <= threshold:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        s_scale = random.uniform(0.5, 1.5)
        s = np.clip(s * s_scale, 0, 255).astype(np.uint8)

        hsv_image = np.stack([h, s, v]).transpose((1, 2, 0))
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return image

def random_gray(image, threshold = GRAY):
    if random.random() <= threshold:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.stack([image, image, image]).transpose((1, 2, 0))

    return image

def random_gaussian_noise(image, sigma = 2, threshold = GAUSSIAN_NOISE):
    if random.random() <= threshold:
        sigma = random.uniform(0, sigma)
        image = cv2.GaussianBlur(image, (5, 5), sigmaX = sigma)
    
    return image

def random_shift(image, gt_bboxes = None, mean = 128, threshold = SHIFT):
    if random.random() <= threshold:
        margin = random.uniform(1.0, 1.5)
        h, w, c = image.shape

        expand_w = int(margin * w)
        expand_h = int(margin * h)

        left, top, right, down = 0, 0, 0, 0
        
        if random.choice([True, False]):
            left = random.randint(0, expand_w - w)

        if random.choice([True, False]):
            top = random.randint(0, expand_h - h)

        if random.choice([True, False]):
            right = random.randint(0, expand_w - w)

        if random.choice([True, False]):
            down = random.randint(0, expand_h - h)

        shift_image = np.full((h + top + down, w + left + right, 3), mean, dtype = np.uint8)
        shift_image[top:top + h, left:left + w, :] = image.copy()

        image = shift_image
        
        if gt_bboxes is not None:
            gt_bboxes[:, 0] += left
            gt_bboxes[:, 1] += top
            gt_bboxes[:, 2] += left
            gt_bboxes[:, 3] += top

    if gt_bboxes is not None:
        return image, gt_bboxes.astype(np.float32)
    else:
        return image

def random_crop(image, gt_bboxes = None, gt_classes = None, threshold = CROP):
    if random.random() <= threshold:
        # select bboxes
        indexs = range(len(gt_bboxes))
        indexs = random.sample(indexs, k = random.randint(1, len(indexs)))
        
        # select max bbox
        max_x1y1 = np.min(gt_bboxes[indexs, :2], axis = 0)
        max_x2y2 = np.max(gt_bboxes[indexs, 2:], axis = 0)
        max_bbox = np.concatenate([max_x1y1, max_x2y2], axis = 0)
        
        # margin (left, top, right, down)
        h, w, c = image.shape
        max_l_trans = max_bbox[0]
        max_t_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        # random crop
        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_t_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        # update max bbox
        max_bbox = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]

        # select mask
        cx_list = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        cy_list = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        center_list = np.stack([cx_list, cy_list]).T

        x_mask = np.logical_and(max_bbox[0] <= center_list[:, 0], center_list[:, 0] <= max_bbox[2])
        y_mask = np.logical_and(max_bbox[1] <= center_list[:, 1], center_list[:, 1] <= max_bbox[3])
        mask = np.logical_and(x_mask, y_mask)

        # crop image, update gt_bboxes & gt_classes
        gt_bboxes = gt_bboxes[mask]
        gt_classes = gt_classes[mask]

        image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
        h, w, c = image.shape

        gt_bboxes[:, [0, 2]] = np.clip(gt_bboxes[:, [0, 2]] - crop_xmin, 0, w - 1)
        gt_bboxes[:, [1, 3]] = np.clip(gt_bboxes[:, [1, 3]] - crop_ymin, 0, h - 1)

    return image, gt_bboxes.astype(np.float32), gt_classes

'''
# Focal Loss for Dense Object Detection
4.1 Inference and Training - Optimization
We use horizontal image flipping as the only form of data augmentation unless otherwise noted.
'''
def DataAugmentation(image, gt_bboxes, gt_classes):
    # image = random_hue(image)
    # image = random_saturation(image)
    # image = random_gray(image)
    # image = random_brightness(image)
    image = random_gaussian_noise(image)
    
    # image, gt_bboxes = random_scale(image, gt_bboxes)
    # image, gt_bboxes = random_shift(image, gt_bboxes)
    # image, gt_bboxes = random_vertical_flip(image, gt_bboxes)
    image, gt_bboxes = random_horizontal_flip(image, gt_bboxes)
    
    image, gt_bboxes, gt_classes = random_crop(image, gt_bboxes, gt_classes)
    
    return image.astype(np.uint8), gt_bboxes, gt_classes
