# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import tensorflow as tf

from Define import *

'''
pt = {
    p    , if y = 1
    1 − p, otherwise
}
FL(pt) = −(1 − pt)γ * log(pt)
'''
def Focal_Loss(pred_classes, gt_classes, alpha = 0.25, gamma = 2):
    with tf.variable_scope('Focal'):
        # positive_mask = [BATCH_SIZE, 22890]
        positive_mask = tf.reduce_max(gt_classes[:, :, 1:], axis = -1)
        positive_mask = tf.cast(tf.math.equal(positive_mask, 1.), dtype = tf.float32)

        ignored_mask = tf.cast(tf.math.equal(gt_classes[:, :, 0], -1), dtype = tf.float32) # -1 == 1
        ignored_mask = tf.expand_dims(1 - ignored_mask, axis = -1) # 1 -> 0
        
        # exception 1. log(-1) -> nan !
        gt_classes = ignored_mask * gt_classes
        pred_classes = ignored_mask * pred_classes
        
        # positive_count = [BATCH_SIZE]
        positive_count = tf.reduce_sum(positive_mask, axis = 1)
        positive_count = tf.clip_by_value(positive_count, 1, positive_count)

        # focal_loss = [BATCH_SIZE, 22890, CLASSES]
        pt = gt_classes * pred_classes + (1 - gt_classes) * (1 - pred_classes) 
        focal_loss = -alpha * tf.pow(1. - pt, gamma) * tf.log(pt + 1e-10)

        # focal_loss = [BATCH_SIZE]
        focal_loss = tf.reduce_sum(ignored_mask * tf.abs(focal_loss), axis = [1, 2])
        focal_loss = tf.reduce_mean(focal_loss / positive_count)

    return focal_loss

'''
GIoU = IoU - (C - (A U B))/C
Loss = 1 - GIoU
'''
def GIoU(bboxes_1, bboxes_2):
    with tf.variable_scope('GIoU'):
        # 1. calulate intersection over union
        area_1 = (bboxes_1[..., 2] - bboxes_1[..., 0]) * (bboxes_1[..., 3] - bboxes_1[..., 1])
        area_2 = (bboxes_2[..., 2] - bboxes_2[..., 0]) * (bboxes_2[..., 3] - bboxes_2[..., 1])
        
        intersection_wh = tf.minimum(bboxes_1[:, :, 2:], bboxes_2[:, :, 2:]) - tf.maximum(bboxes_1[:, :, :2], bboxes_2[:, :, :2])
        intersection_wh = tf.maximum(intersection_wh, 0)
        
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        union = (area_1 + area_2) - intersection
        
        ious = intersection / tf.maximum(union, 1e-10)

        # 2. (C - (A U B))/C
        C_wh = tf.maximum(bboxes_1[..., 2:], bboxes_2[..., 2:]) - tf.minimum(bboxes_1[..., :2], bboxes_2[..., :2])
        C_wh = tf.maximum(C_wh, 0.0)
        C = C_wh[..., 0] * C_wh[..., 1]
        
        giou = ious - (C - union) / tf.maximum(C, 1e-10)
    return giou

def RetinaNet_Loss(pred_bboxes, pred_classes, gt_bboxes, gt_classes, alpha = 0.25):
    # calculate focal_loss & GIoU_loss
    focal_loss_op = Focal_Loss(pred_classes, gt_classes)
    
    # positive_mask = [BATCH_SIZE, 22890]
    positive_mask = tf.reduce_max(gt_classes[:, :, 1:], axis = -1)
    positive_mask = tf.cast(tf.math.equal(positive_mask, 1.), dtype = tf.float32)
    
    # positive_count = [BATCH_SIZE]
    positive_count = tf.reduce_sum(positive_mask, axis = 1)
    positive_count = tf.clip_by_value(positive_count, 1, positive_count)
    
    giou_loss_op = 1 - GIoU(pred_bboxes, gt_bboxes)

    giou_loss_op = tf.reduce_sum(positive_mask * giou_loss_op, axis = 1)
    giou_loss_op = tf.reduce_mean(giou_loss_op / positive_count)
    
    loss_op = focal_loss_op + alpha * giou_loss_op
    
    return loss_op, focal_loss_op, giou_loss_op

if __name__ == '__main__':
    ## check loss shape
    pred_bboxes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, 4])
    pred_classes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, CLASSES])

    gt_bboxes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, 4])
    gt_classes = tf.placeholder(tf.float32, [BATCH_SIZE, 22890, CLASSES])

    loss_op, focal_loss_op, smooth_l1_loss_op = RetinaNet_Loss(pred_bboxes, pred_classes, gt_bboxes, gt_classes)
    print(loss_op, focal_loss_op, smooth_l1_loss_op)

    ## check ignored mask
    # import numpy as np
    # gt_classes = np.zeros((1, 5, 5))
    # gt_classes[:, :, 0] = -1
    # gt_classes[:, 0, 0] = 0
    # gt_classes[:, 1, 0] = 0

    # ignored_mask = tf.cast(tf.math.equal(gt_classes[:, :, 0], -1), dtype = tf.float32) # -1 == 1
    # ignored_mask = tf.expand_dims(1 - ignored_mask, axis = -1) # 1 -> 0
    
    # sess = tf.Session()
    # print(sess.run(ignored_mask))