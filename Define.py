# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

# dataset parameters
ROOT_DIR = 'D:/_ImageDataset/Wider_Face_Dataset/'

CLASS_NAMES = ['background'] + ['Face']
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}
CLASSES = len(CLASS_NAMES)

# network parameters
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_CHANNEL = 3

SAMPLE_IMAGE_HEIGHT = 224
SAMPLE_IMAGE_WIDTH = 224

R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94
MEAN = [R_MEAN, G_MEAN, B_MEAN]

PYRAMID_LEVELS = [2, 3, 4, 5, 6]

ANCHOR_SCALES = [2 ** 0, 2 ** (1/3), 2 ** (2/3)] # [1, 1.2599210498948732, 1.5874010519681994]
ANCHOR_SIZES = [2 ** (x + 2) for x in PYRAMID_LEVELS]

ANCHORS = len(ANCHOR_SCALES)

AP_THRESHOLD = 0.5
NMS_THRESHOLD = 0.6

POSITIVE_IOU_THRESHOLD = 0.5
NEGATIVE_IOU_THRESHOLD = 0.3

# loss parameters
WEIGHT_DECAY = 0.0005

# train
# use thread (Dataset)
NUM_THREADS = 10

# single gpu training
GPU_INFO = "0,1,2,3"
NUM_GPU = len(GPU_INFO.split(','))

BATCH_SIZE = 4 * NUM_GPU
INIT_LEARNING_RATE = 1e-4

# iteration & learning rate schedule
MAX_ITERATION = 200000
DECAY_ITERATIONS = [100000, 160000]

SAMPLES = BATCH_SIZE

LOG_ITERATION = 50
SAMPLE_ITERATION = 5000
VALID_ITERATION = 5000

# color_list (OpenCV - BGR)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

COLOR_PBLUE = (204, 72, 63)
COLOR_ORANGE = (0, 128, 255)
