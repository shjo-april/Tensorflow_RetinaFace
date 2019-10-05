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
from Teacher import *

from RetinaNet import *
from RetinaNet_Loss import *
from RetinaNet_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INFO

# 1. dataset
train_data_list = np.load('./dataset/train.npy', allow_pickle = True)
valid_data_list = np.load('./dataset/validation.npy', allow_pickle = True)
valid_count = len(valid_data_list)

open('log.txt', 'w')
log_print('[i] Train : {}'.format(len(train_data_list)))
log_print('[i] Valid : {}'.format(len(valid_data_list)))

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
is_training = tf.placeholder(tf.bool)

input_vars = tf.split(input_var, NUM_GPU)

pred_bboxes_ops = []
pred_classes_ops = []

for gpu_id in range(NUM_GPU):
    reuse = gpu_id != 0
    with tf.device(tf.DeviceSpec(device_type = "GPU", device_index = gpu_id)):
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            print(input_vars[gpu_id], is_training, reuse)
            retina_dic, retina_sizes = RetinaNet(input_vars[gpu_id], is_training)

            if not reuse:
                retina_utils = RetinaNet_Utils()
            
            pred_bboxes_ops.append(retina_dic['pred_bboxes'])
            pred_classes_ops.append(retina_dic['pred_classes'])

pred_bboxes_op = tf.concat(pred_bboxes_ops, axis = 0)
pred_classes_op = tf.concat(pred_classes_ops, axis = 0)

retina_utils.generate_anchors(retina_sizes)
pred_bboxes_op = Decode_Layer(retina_dic['pred_bboxes'], retina_utils.anchors)

_, retina_size, _ = pred_bboxes_op.shape.as_list()
gt_bboxes_var = tf.placeholder(tf.float32, [None, retina_size, 4])
gt_classes_var = tf.placeholder(tf.float32, [None, retina_size, CLASSES])

log_print('[i] pred_bboxes_op : {}'.format(pred_bboxes_op))
log_print('[i] pred_classes_op : {}'.format(pred_classes_op))
log_print('[i] gt_bboxes_var : {}'.format(gt_bboxes_var))
log_print('[i] gt_classes_var : {}'.format(gt_classes_var))

loss_op, focal_loss_op, giou_loss_op = RetinaNet_Loss(pred_bboxes_op, pred_classes_op, gt_bboxes_var, gt_classes_var)

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op, colocate_gradients_with_ops = True)

train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/Focal_Loss' : focal_loss_op,
    'Loss/GIoU_Loss' : giou_loss_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op,
    'Learning_rate' : learning_rate_var,
}

train_summary_list = []
for name in train_summary_dic.keys():
    value = train_summary_dic[name]
    train_summary_list.append(tf.summary.scalar(name, value))
train_summary_op = tf.summary.merge(train_summary_list)

log_image_var = tf.placeholder(tf.float32, [None, SAMPLE_IMAGE_HEIGHT, SAMPLE_IMAGE_WIDTH, IMAGE_CHANNEL])
log_image_op = tf.summary.image('Image/Train', log_image_var[..., ::-1], SAMPLES)

# 3. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# '''
pretrained_vars = []
for var in vars:
    if 'resnet_v1_50' in var.name:
        pretrained_vars.append(var)

pretrained_saver = tf.train.Saver(var_list = pretrained_vars)
pretrained_saver.restore(sess, './resnet_v1_model/resnet_v1_50.ckpt')
# '''

saver = tf.train.Saver(max_to_keep = 10)
# saver.restore(sess, './model/RetinaNet_{}.ckpt'.format(30000))

learning_rate = INIT_LEARNING_RATE

log_print('[i] max_iteration : {}'.format(MAX_ITERATION))
log_print('[i] decay_iteration : {}'.format(DECAY_ITERATIONS))

best_valid_loss = 10.0

loss_list = []
focal_loss_list = []
giou_loss_list = []
l2_reg_loss_list = []
train_time = time.time()

train_writer = tf.summary.FileWriter('./logs/train')

# generate validation data
valid_image_data = []
valid_encode_bboxes = []
valid_encode_classes = []

for i, data in enumerate(valid_data_list):
    image_name, gt_bboxes, gt_classes = data
                
    image_path = ROOT_DIR + 'validation/' + image_name
    gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32)
    gt_classes = np.asarray([1 for c in gt_classes], dtype = np.int32)

    if len(gt_bboxes) == 0:
        continue
    
    image = cv2.imread(image_path)
    image_h, image_w, image_c = image.shape

    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

    gt_bboxes = gt_bboxes.astype(np.float32)
    gt_bboxes /= [image_w, image_h, image_w, image_h]
    gt_bboxes *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]

    encode_bboxes, encode_classes = retina_utils.Encode(gt_bboxes, gt_classes)

    valid_image_data.append(image.astype(np.float32))
    valid_encode_bboxes.append(encode_bboxes)
    valid_encode_classes.append(encode_classes)

    sys.stdout.write('\r[{}/{}]'.format(i, valid_count))
    sys.stdout.flush()

log_print(' [i] generate validation data')

train_threads = []
for i in range(NUM_THREADS):
    train_thread = Teacher(train_data_list, retina_sizes, debug = False)
    train_thread.start()
    train_threads.append(train_thread)

sample_data_list = train_data_list[:SAMPLES]

for iter in range(1, MAX_ITERATION + 1):
    if iter in DECAY_ITERATIONS:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))

    # Thread
    find = False
    while not find:
        for train_thread in train_threads:
            if train_thread.ready:
                find = True
                batch_image_data, batch_encode_bboxes, batch_encode_classes = train_thread.get_batch_data()        
                break
                
    _feed_dict = {input_var : batch_image_data, gt_bboxes_var : batch_encode_bboxes, gt_classes_var : batch_encode_classes, is_training : True, learning_rate_var : learning_rate}
    log = sess.run([train_op, loss_op, focal_loss_op, giou_loss_op, l2_reg_loss_op, train_summary_op], feed_dict = _feed_dict)
    # print(log[1:-1])
    
    if np.isnan(log[1]):
        print('[!]', log[1:-1])
        input()

    loss_list.append(log[1])
    focal_loss_list.append(log[2])
    giou_loss_list.append(log[3])
    l2_reg_loss_list.append(log[4])
    train_writer.add_summary(log[5], iter)
    
    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        focal_loss = np.mean(focal_loss_list)
        giou_loss = np.mean(giou_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter : {}, loss : {:.4f}, focal_loss : {:.4f}, giou_loss : {:.4f}, l2_reg_loss : {:.4f}, train_time : {}sec'.format(iter, loss, focal_loss, giou_loss, l2_reg_loss, train_time))

        loss_list = []
        focal_loss_list = []
        giou_loss_list = []
        l2_reg_loss_list = []
        train_time = time.time()

    if iter % SAMPLE_ITERATION == 0:
        total_gt_bboxes = []
        batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)

        for i, data in enumerate(sample_data_list):
            image_name, gt_bboxes, gt_classes = data

            image_path = ROOT_DIR + image_name

            image = cv2.imread(image_path)
            h, w, c = image.shape
            tf_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

            gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32) / [w, h, w, h] * [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]

            batch_image_data[i] = tf_image.copy()
            total_gt_bboxes.append(gt_bboxes)

        total_pred_bboxes, total_pred_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : batch_image_data, is_training : False})
        
        sample_images = []
        for i in range(BATCH_SIZE):
            image = batch_image_data[i]
            pred_bboxes, pred_classes = retina_utils.Decode(total_pred_bboxes[i], total_pred_classes[i], [IMAGE_WIDTH, IMAGE_HEIGHT], detect_threshold = 0.20)
            
            for bbox, class_index in zip(pred_bboxes, pred_classes):
                xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
                conf = bbox[4]
                class_name = CLASS_NAMES[class_index]

                string = "{} : {:.2f}%".format(class_name, conf * 100)
                cv2.putText(image, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            for bbox in total_gt_bboxes[i]:
                xmin, ymin, xmax, ymax = bbox.astype(np.int32)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            image = cv2.resize(image, (SAMPLE_IMAGE_WIDTH, SAMPLE_IMAGE_HEIGHT))
            sample_images.append(image.copy())
        
        image_summary = sess.run(log_image_op, feed_dict = {log_image_var : sample_images})
        train_writer.add_summary(image_summary, iter)

    if iter % VALID_ITERATION == 0:
        valid_loss_list = []
        valid_length = len(valid_data_list)
        valid_iteration = valid_length // BATCH_SIZE
        
        valid_time = time.time()

        for valid_iter in range(valid_iteration):
            batch_image_data = valid_image_data[valid_iter * BATCH_SIZE : (valid_iter + 1) * BATCH_SIZE]
            batch_encode_bboxes = valid_encode_bboxes[valid_iter * BATCH_SIZE : (valid_iter + 1) * BATCH_SIZE]
            batch_encode_classes = valid_encode_classes[valid_iter * BATCH_SIZE : (valid_iter + 1) * BATCH_SIZE]

            valid_losses = sess.run([focal_loss_op, giou_loss_op], feed_dict = {input_var : batch_image_data, gt_bboxes_var : batch_encode_bboxes, gt_classes_var : batch_encode_classes, is_training : False, })
            valid_loss = np.sum(valid_losses)

            valid_loss_list.append(valid_loss)

            sys.stdout.write('\rvalidation = [{}/{}]'.format(valid_iter, valid_iteration))
            sys.stdout.flush()

        valid_time = int(time.time() - valid_time)
        
        valid_loss = np.mean(valid_loss_list)
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            saver.save(sess, './model/RetinaNet_{}.ckpt'.format(iter))

        print()
        log_print('[i] iter : {}, valid_loss : {:.4f}, best_valid_loss : {:.4f}, valid_time : {}sec'.format(iter, valid_loss, best_valid_loss, valid_time))

saver.save(sess, './model/RetinaNet.ckpt')