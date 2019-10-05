# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import glob
import numpy as np

import xml.etree.ElementTree as ET

from Define import *

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()

def xml_read(xml_path, find_labels = CLASS_NAMES, normalize = False):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_path = xml_path[:-3] + '*'
    image_path = image_path.replace('/xml', '/image')
    image_path = glob.glob(image_path)[0]

    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        if not label in find_labels:
            continue
            
        bbox = obj.find('bndbox')
        
        bbox_xmin = max(min(int(bbox.find('xmin').text.split('.')[0]), image_width - 1), 0)
        bbox_ymin = max(min(int(bbox.find('ymin').text.split('.')[0]), image_height - 1), 0)
        bbox_xmax = max(min(int(bbox.find('xmax').text.split('.')[0]), image_width - 1), 0)
        bbox_ymax = max(min(int(bbox.find('ymax').text.split('.')[0]), image_height - 1), 0)

        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        if normalize:
            bbox = np.asarray([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax], dtype = np.float32)
            bbox /= [image_width, image_height, image_width, image_height]
            bbox *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(CLASS_DIC[label])

    return image_path, np.asarray(bboxes, dtype = np.float32), np.asarray(classes, dtype = np.int32)

def one_hot(gt_class, classes = CLASSES):
    label = np.zeros((classes), dtype = np.float32)
    label[gt_class] = 1.
    return label
    
def convert_bboxes(bboxes, image_wh, ori_wh = [IMAGE_WIDTH, IMAGE_HEIGHT]):
    return bboxes / (ori_wh * 2) * (image_wh * 2)

def compute_bboxes_IoU(bboxes_1, bboxes_2):
    area_1 = (bboxes_1[:, 2] - bboxes_1[:, 0] + 1) * (bboxes_1[:, 3] - bboxes_1[:, 1] + 1)
    area_2 = (bboxes_2[:, 2] - bboxes_2[:, 0] + 1) * (bboxes_2[:, 3] - bboxes_2[:, 1] + 1)

    iw = np.minimum(bboxes_1[:, 2][:, np.newaxis], bboxes_2[:, 2]) - np.maximum(bboxes_1[:, 0][:, np.newaxis], bboxes_2[:, 0]) + 1
    ih = np.minimum(bboxes_1[:, 3][:, np.newaxis], bboxes_2[:, 3]) - np.maximum(bboxes_1[:, 1][:, np.newaxis], bboxes_2[:, 1]) + 1

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    
    intersection = iw * ih
    union = (area_1[:, np.newaxis] + area_2) - iw * ih

    return intersection / np.maximum(union, 1e-10)

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def class_nms(pred_bboxes, pred_classes, threshold = NMS_THRESHOLD):
    data_dic = {}
    nms_bboxes = []
    nms_classes = []

    for bbox, class_index in zip(pred_bboxes, pred_classes):
        try:
            data_dic[class_index].append(bbox)
        except KeyError:
            data_dic[class_index] = []
            data_dic[class_index].append(bbox)
    
    for key in data_dic.keys():
        pred_bboxes = np.asarray(data_dic[key], dtype = np.float32)
        pred_bboxes = nms(pred_bboxes, threshold)

        for pred_bbox in pred_bboxes:
            nms_bboxes.append(pred_bbox)
            nms_classes.append(key)
    
    return np.asarray(nms_bboxes, dtype = np.float32), np.asarray(nms_classes, dtype = np.int32)
