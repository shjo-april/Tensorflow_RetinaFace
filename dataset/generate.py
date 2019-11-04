import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description = 'DB', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_dir', dest='image_dir', help="", default='', type = str)
    parser.add_argument('--txt_path', dest='txt_path', help="", default='', type = str)
    parser.add_argument('--npy_path', dest='npy_path', help="", type = str)
    args = parser.parse_args()
    return args

args = parse_args()

image_dir = args.image_dir
txt_path = args.txt_path

with open(txt_path, 'r') as f:
    lines = f.readlines()

data_list = []

i = 0
line_length = len(lines)

while i < line_length:
    image_name = lines[i].strip(); i += 1
    gt_bboxes = []
    gt_classes = []

    bbox_count = int(lines[i].strip()); i += 1
    if bbox_count == 0:
        while not '.jpg' in lines[i]:
            i += 1
        continue
        
    for _ in range(bbox_count):
        bbox_info = lines[i].strip().split(' '); i += 1

        xmin, ymin, w, h = np.asarray(bbox_info[:4], dtype = np.int32)
        xmax = xmin + w
        ymax = ymin + h

        gt_bboxes.append([xmin, ymin, xmax, ymax])
        gt_classes.append('Face')
    
    # image = cv2.imread(image_dir + image_name)

    # for bbox in gt_bboxes:
    #     xmin, ymin, xmax, ymax = bbox
    #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # cv2.imshow('show', image)
    # cv2.waitKey(0)

    data_list.append([image_name, gt_bboxes, gt_classes])

np.save(args.npy_path, data_list)
