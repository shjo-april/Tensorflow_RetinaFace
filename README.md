# RetinaFace: Single-stage Dense Face Localisation in the Wild

## # Difference from Paper
1. standard OHEM -> Focal Loss
2. Smooth L1 Loss -> GIoU
3. Momentum Optimizer -> Adam Optimizer

## # Reference
- Focal Loss for Dense Object Detection [[Paper]](https://arxiv.org/abs/1708.02002)
- RetinaFace: Single-stage Dense Face Localisation in the Wild [[Paper]](https://arxiv.org/abs/1905.00641)
- Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.pdf) [[Code]](https://github.com/OFRIN/Tensorflow_GIoU)
