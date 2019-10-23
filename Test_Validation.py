# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import glob
import time
import random

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *
from DataAugmentation import *

from RetinaFace import *
from RetinaFace_Loss import *
from RetinaFace_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 1. dataset
test_data_list = np.load('./dataset/validation.npy', allow_pickle = True)

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

retina_dic, retina_sizes = RetinaFace(input_var, False)

retina_utils = RetinaFace_Utils()
retina_utils.generate_anchors(retina_sizes)

pred_bboxes_op = Decode_Layer(retina_dic['pred_bboxes'], retina_utils.anchors)
pred_classes_op = retina_dic['pred_classes']

# 3. test
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
# saver.restore(sess, './model/RetinaFace_{}.ckpt'.format(130000))
saver.restore(sess, './model/RetinaFace.ckpt')

batch_size = 2

for test_iter in range(len(test_data_list) // batch_size):
    total_gt_bboxes = []
    batch_data_list = test_data_list[test_iter * batch_size : (test_iter + 1) * batch_size]
    batch_image_data = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)

    for i, data in enumerate(batch_data_list):
        image_name, gt_bboxes, gt_classes = data
        image_path = ROOT_DIR + 'validation/' + image_name

        gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32)

        image = cv2.imread(image_path)
        image_h, image_w, c = image.shape

        gt_bboxes /= [image_w, image_h, image_w, image_h]
        gt_bboxes *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]

        tf_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

        batch_image_data[i] = tf_image.copy()
        total_gt_bboxes.append(gt_bboxes)

    total_pred_bboxes, total_pred_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : batch_image_data})

    for i in range(batch_size):
        image = batch_image_data[i]
        pred_bboxes, pred_classes = retina_utils.Decode(total_pred_bboxes[i], total_pred_classes[i], [IMAGE_WIDTH, IMAGE_HEIGHT], detect_threshold = 0.50)
        
        for bbox, class_index in zip(pred_bboxes, pred_classes):
            xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
            conf = bbox[4]
            class_name = CLASS_NAMES[class_index]
            
            # string = "{} : {:.2f}%".format(class_name, conf * 100)
            # cv2.putText(image, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # for gt_bbox in total_gt_bboxes[i]:
        #     xmin, ymin, xmax, ymax = gt_bbox.astype(np.int32)
        #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        cv2.imshow('show', image.astype(np.uint8))
        cv2.waitKey(0)
