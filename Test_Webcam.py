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

# 1. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

retina_dic, retina_sizes = RetinaFace(input_var, False)

retina_utils = RetinaFace_Utils()
retina_utils.generate_anchors(retina_sizes)

pred_bboxes_op = Decode_Layer(retina_dic['pred_bboxes'], retina_utils.anchors)
pred_classes_op = retina_dic['pred_classes']

# 2. test
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/RetinaFace.ckpt')

video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    if not ret:
        break

    h, w, c = frame.shape

    tf_image = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
    total_pred_bboxes, total_pred_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : [tf_image.astype(np.float32)]})

    pred_bboxes, pred_classes = retina_utils.Decode(total_pred_bboxes[0], total_pred_classes[0], [w, h], detect_threshold = 0.50)
    
    for bbox, class_index in zip(pred_bboxes, pred_classes):
        xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
        conf = bbox[4]
        class_name = CLASS_NAMES[class_index]

        string = "{} : {:.2f}%".format(class_name, conf * 100)
        cv2.putText(frame, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow('show', frame)
    cv2.waitKey(1)
