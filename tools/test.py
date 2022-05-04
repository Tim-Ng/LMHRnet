import torch 
from PIL import Image, ImageOps
from torchvision import transforms
from munch import DefaultMunch
from tensorboardX import SummaryWriter
import time
import argparse
import os
import pprint
import shutil
import sys
from tqdm import tqdm
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
os.chdir("tools")
import _init_paths
import models
import datasets
from src import _evaluator
from config import config
from config import update_config
from datasets import CustlrDataset
import cv2
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pandas as pd
os.chdir("../")
            
def testing(cfg,pretrain_model,disply_result=True,other_testfile= None):
    update_config(config, cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if other_testfile is None:
        df = pd.read_csv(config.DATASET.TEST_SET)
    else:
        df = pd.read_csv(other_testfile)
    df = df.iloc[:6, :]
    dataset = CustlrDataset(df,config, mode='NOCROP',base_path=config.DATASET.ROOT)
    print('dataset', len(dataset))
    #dataset.plot_landmark_map(0)
    #dataset.plot_sample(0)
    
    ldr = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    EPOCH = 1
    val_pos_loss = []

    # when load empty model, make sure to set the number of landmark maps as the output channel of the model
    out_channel = dataset[0]['landmark_map'].shape[0]

    evaluator = _evaluator(config,out_channel)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load empty model
    # build model
    module = eval('models.'+config.MODEL.NAME)
    module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +'.get_seg_body_model')(config)
    statedict = torch.load(pretrain_model)

    val_pos_loss = []
    # load model state
    model.load_state_dict(statedict)
    for i, param in enumerate(model.parameters()):
        param.requires_grad = False

    model.eval()
    model.to(device)

    for e in range(EPOCH):
        for i, sample in enumerate(ldr):
            for key in sample:
                sample[key] = sample[key].to(device)
            output = model(sample)
            evaluator.add(output, sample)
            if (disply_result):
                print(output['lm_pos_output'][0])
                dataset.plot_sample(i)
                 # body landmarks
                plt.scatter(output['lm_pos_output'][0,:,0]*224,output['lm_pos_output'][0,:,1]*224, s=5, color='red')
                predname = 'pred_visualized/gt/test/'+'img'+str(i)+'.jpg'
                # plt.savefig(predname)
                plt.show()

                landmark_map = np.max(output['lm_pos_map'][0].cpu().numpy(), axis=0)
                plt.imshow(landmark_map)
                predname = 'pred_visualized/alt/v3_mine-v-gdrive/'+'hm'+str(i)+'.jpg'
                # plt.savefig(predname)

        ret = evaluator.evaluate()
        val_pos_loss.append(ret['lm_dist'])
    # show avg losses
    print(val_pos_loss)
    print(ret['lm_individual_dist'])

    
def testing_scratch(cfg,pretrain_model,disply_result=True,other_testfile= None):
    update_config(config, cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if other_testfile is None:
        df = pd.read_csv(config.DATASET.TEST_SET)
    else:
        df = pd.read_csv(other_testfile)
    df = df.iloc[:6, :]
    dataset = CustlrDataset(df,config, mode='NOCROP',base_path=config.DATASET.ROOT)
    print('dataset', len(dataset))
    #dataset.plot_landmark_map(0)
    #dataset.plot_sample(0)
    
    ldr = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    EPOCH = 1
    val_pos_loss = []

    # when load empty model, make sure to set the number of landmark maps as the output channel of the model
    out_channel = dataset[0]['landmark_map'].shape[0]

    evaluator = _evaluator(config,out_channel)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load empty model
    # build model
    module = eval('models.'+config.MODEL.NAME)
    module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +'.get_body_model')(config)
    statedict = torch.load(pretrain_model)

    val_pos_loss = []
    # load model state
    model.load_state_dict(statedict)
    for i, param in enumerate(model.parameters()):
        param.requires_grad = False

    model.eval()
    model.to(device)

    for e in range(EPOCH):
        for i, sample in enumerate(ldr):
            for key in sample:
                sample[key] = sample[key].to(device)
            output = model(sample)
            evaluator.add(output, sample)
            if (disply_result):
                print(output['lm_pos_output'][0])
                dataset.plot_sample(i)
                 # body landmarks
                plt.scatter(output['lm_pos_output'][0,:,0]*224,output['lm_pos_output'][0,:,1]*224, s=5, color='red')
                predname = 'pred_visualized/gt/test/'+'img'+str(i)+'.jpg'
                # plt.savefig(predname)
                plt.show()

                landmark_map = np.max(output['lm_pos_map'][0].cpu().numpy(), axis=0)
                plt.imshow(landmark_map)
                predname = 'pred_visualized/alt/v3_mine-v-gdrive/'+'hm'+str(i)+'.jpg'
                # plt.savefig(predname)

        ret = evaluator.evaluate()
        val_pos_loss.append(ret['lm_dist'])
    # show avg losses
    print(val_pos_loss)
    print(ret['lm_individual_dist'])