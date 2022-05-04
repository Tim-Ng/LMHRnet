
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.NUM_OUTPUTS = 1
_C.MODEL.EXTRA = CN(new_allowed=True)


_C.MODEL.OCR = CN()
_C.MODEL.OCR.MID_CHANNELS = 512
_C.MODEL.OCR.KEY_CHANNELS = 256
_C.MODEL.OCR.DROPOUT = 0.05
_C.MODEL.OCR.SCALE = 1

_C.LOSS = CN()
_C.LOSS.WEIGHT_LOSS_CATEGORY = 1
_C.LOSS.WEIGHT_LOSS_ATTR = 20
_C.LOSS.WEIGHT_LOSS_LM_POS = 100
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'lip'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.NUM_CLASSES_BDLM = 12
_C.DATASET.TRAIN_SET = 'list/lip/train.lst'
_C.DATASET.VAL_SET = 'list/lip/val.lst'
_C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'list/lip/test.lst'

# training
_C.TRAIN = CN()

_C.TRAIN.FREEZE_LAYERS = ''
_C.TRAIN.FREEZE_EPOCHS = -1
_C.TRAIN.NONBACKBONE_KEYWORDS = []
_C.TRAIN.NONBACKBONE_MULT = 10

_C.TRAIN.IMAGE_SIZE = [1024, 512]  # width * height
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16

_C.TRAIN.RANDOM_BRIGHTNESS = False
_C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
# only using some training samples
_C.TRAIN.NUM_SAMPLES = 0
_C.TRAIN.LEARNING_RATE_DECAY = 0.8
_C.TRAIN.WEIGHT_LOSS_LM_POS = 10

# testing
_C.TEST = CN()

_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048

_C.TEST.BATCH_SIZE_PER_GPU = 32
# only testing some samples
_C.TEST.NUM_SAMPLES = 0

_C.TEST.MODEL_FILE = '../pretrained_models/hrnet_w48_lip_cls20_473x473.pth'
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False
_C.TEST.SCALE_LIST = [1]

_C.TEST.OUTPUT_INDEX = -1

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False

#constant for body landmark
_C.CONSTANT = CN()
_C.CONSTANT.LM_TRAIN_USE = "in pic"
_C.CONSTANT.LM_EVAL_USE = "in pic"
_C.CONSTANT.lm2name = []
_C.CONSTANT.IMAGE_SIZE = [224,224]
_C.CONSTANT.RESIZE = 256
_C.CONSTANT.NAME_DIV8 = 'landmark_map28'
_C.CONSTANT.NAME_DIV4 = 'landmark_map56'
_C.CONSTANT.NAME_DIV2 = 'landmark_map112'
_C.CONSTANT.NAME_NODIV= 'landmark_map224'
_C.CONSTANT.SAVEFILE = 'pretrained models'

#def update_config(cfg, args)
def update_config(cfg, cfg_file_path):
    cfg.defrost()
    
    #cfg.merge_from_file(args.cfg) if using arg
    cfg.merge_from_file(cfg_file_path)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

