import cv2
import argparse
import numpy as np

image_dir = 'D:/_DeepLearning_DB/Wider_Face_Dataset/train/'
data_list = np.load('./dataset/train.npy', allow_pickle = True)

for data in data_list:
    image_name, gt_bboxes, gt_classes = data

    image_path = image_dir + image_name

    image = cv2.imread(image_path)

    for bbox in gt_bboxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow('show', image)
    cv2.waitKey(0)
