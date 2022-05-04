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

def training(cfg,learn=0.001,NUMEPOCH = 20):
    learning_rate = learn
    update_config(config, cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    new_model_name = 'lm param epoch='+str(NUMEPOCH)+' lr =' + str(learn) +' size=' + str(config.CONSTANT.IMAGE_SIZE[0]) + ' hl='+ str(config.MODEL.EXTRA['STAGE5']['NUM_MODULES']) + ' 8960 dataset'
    # build model
    #module = eval('models.seg_hrnet_bodylandmark')
    module = eval('models.'+config.MODEL.NAME)
    module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    #model = eval('models.seg_hrnet_bodylandmark.get_seg_body_model')(config)
    model = eval('models.'+config.MODEL.NAME +'.get_seg_body_model')(config)
    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    pretrained_dict = torch.load(config.MODEL.PRETRAINED)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    print("Loading model...")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    model.to("cuda")
    #load dataset
    traincsv=config.DATASET.TRAIN_SET
    valcsv=config.DATASET.VAL_SET
    base_path = config.DATASET.ROOT
    traindf = pd.read_csv(traincsv)
    valdf = pd.read_csv(valcsv)

    train_dataset = CustlrDataset(traindf,config, mode='NOCROP',base_path = base_path)
    val_dataset = CustlrDataset(valdf,config, mode='NOCROP',base_path = base_path)
    out_channel = train_dataset[0]['landmark_map'].shape[0]
    print("Number of out channels " +str(out_channel))
    evaluator = _evaluator(config,out_channel)

    print('training set', len(train_dataset))
    print('val set', len(val_dataset))
    dssize = len(train_dataset)+len(val_dataset)

    trainloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
    valloader = DataLoader(val_dataset, batch_size=10, shuffle=True, num_workers=0)
    #retrain last layer
    trainstartparam=921
    #freeze some layer params total 0-51 vgg16 extractor 0-25, lm branch upsample 26-51
    for i, param in enumerate(model.parameters()):
        if i >= trainstartparam:
            break
        param.requires_grad = False

    k=trainstartparam
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name:
                torch.nn.init.constant_(param, 0) # initialise bias with zeros
            else:
                #print(name)
                torch.nn.init.uniform_(param) #initialise linear weights with Kaiming/Xavier uniform weights
            print(k,name)
            k=k+1
    logname = time.strftime('%m-%d %H-%M-%S', time.localtime())
    logname = logname + ' param ' + str(trainstartparam)+'-51' + 'dataset size '+ str(dssize)
    logname = 'runs/'+logname
    writer = SummaryWriter(logname)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    train_pos_loss = []
    val_pos_loss = []
    NUMSTEP = len(trainloader)
    step = 0
    for epoch in range(NUMEPOCH):
        for stat in ['train','val']:
            #print(stat)
            if stat == 'train':
                print("---Starting traing---")
                model.train()
                for i, sample in enumerate(trainloader):
                    step += 1
                    for key in sample:
                        sample[key] = sample[key].to("cuda")
                    output = model(sample)
                    #print(output['lm_pos_map'])

                    loss = model.cal_loss(sample, output)
                    optimizer.zero_grad()
                    loss['all'].backward()
                    optimizer.step()

                    if 'lm_vis_loss' in loss:
                        writer.add_scalar('loss/lm_vis_loss', loss['lm_vis_loss'], step)
                        writer.add_scalar('loss_weighted/lm_vis_loss', loss['weighted_lm_vis_loss'], step)
                    if 'lm_pos_loss' in loss:
                        writer.add_scalar('loss/lm_pos_loss', loss['lm_pos_loss'], step)
                        writer.add_scalar('loss_weighted/lm_pos_loss', loss['weighted_lm_pos_loss'], step)
                    writer.add_scalar('loss_weighted/all', loss['all'], step)
                    writer.add_scalar('global/learning_rate', learning_rate, step)

                    train_pos_loss.append(loss['lm_pos_loss'])
                    print('EPOCH:', epoch+1,'/',NUMEPOCH,'step:', i+1,'/',NUMSTEP,'| lm position loss:', loss['lm_pos_loss'],' | weighted loss:', loss['weighted_lm_pos_loss'])
                print("---End traing---")
            else:
                print("---Start evaluation---")
                model.eval()
                for i, sample in enumerate(tqdm(valloader)):
                    step += 1
                    for key in sample:
                        sample[key] = sample[key].to(device)
                    output = model(sample)
                    #print(output['lm_pos_map'])
                    evaluator.add(output, sample)
                ret = evaluator.evaluate()
                for i in range(len(config.CONSTANT.lm2name)):
                    print('metrics/dist_part_{}_{}'.format(i, config.CONSTANT.lm2name[i]), ret['lm_individual_dist'][i])
                    writer.add_scalar('metrics/dist_part_{}_{}'.format(i, config.CONSTANT.lm2name[i]), ret['lm_individual_dist'][i], step)
                print('metrics/dist_all', ret['lm_dist'])
                writer.add_scalar('metrics/dist_all', ret['lm_dist'], step)
                val_pos_loss.append(ret['lm_dist'])
                print('EPOCH:', epoch+1,'/',NUMEPOCH,'| lm distance:', ret['lm_dist'])
                model.train()
                print("---End evaluation---")
        LEARNING_RATE_DECAY =0.9
        learning_rate *= LEARNING_RATE_DECAY
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    try:
        model_file_name=config.MODEL.NAME+' '+new_model_name +' '+ str(trainstartparam) + '-78'+' dataset '+str(dssize) +'.pkl'
        torch.save(model.state_dict(), './'+config.CONSTANT.SAVEFILE+'/' + model_file_name)
        print("MODEL SAVED:"+model_file_name)
    except Exception as e:
        print("AN ERROR HAS OCCURRED WHILE SAVING MODEL:"+str(e))
        
    
    plt.plot(train_pos_loss,label='training pos loss')
    
    tfigname = './'+config.CONSTANT.SAVEFILE+'/'+'experiment layers graph/training losslm/training loss '+new_model_name+' '+str(trainstartparam)+' -78'+' dataset '+str(dssize)+' ' +config.MODEL.NAME+'.png'
    plt.savefig(tfigname)
    plt.show()
    
    plt.plot(val_pos_loss,label='validation pos loss')
    vfigname = './'+config.CONSTANT.SAVEFILE+'/'+ 'experiment layers graph/val losslm/val loss '+new_model_name +' '+str(trainstartparam)+' -78'+' dataset '+str(dssize)+' ' +config.MODEL.NAME+'.png'
    plt.savefig(vfigname)
    plt.show()
    

    
def training_from_scratch(cfg,pretrained=False,learn=0.001,NUMEPOCH = 20):
    learning_rate = learn
    update_config(config, cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    new_model_name = 'lm param epoch='+str(NUMEPOCH)+' lr =' + str(learn) +' size=' + str(config.CONSTANT.IMAGE_SIZE[0]) + ' 8960 dataset'
    # build model
    #module = eval('models.seg_hrnet_bodylandmark')
    module = eval('models.'+config.MODEL.NAME)
    module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    #model = eval('models.seg_hrnet_bodylandmark.get_seg_body_model')(config)
    model = eval('models.'+config.MODEL.NAME +'.get_body_model')(config)
    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    if pretrained:
        pretrained_dict = torch.load(config.MODEL.PRETRAINED)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        print("Loading model...")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.eval()
    model.to("cuda")
    #load dataset
    traincsv=config.DATASET.TRAIN_SET
    valcsv=config.DATASET.VAL_SET
    base_path = config.DATASET.ROOT
    traindf = pd.read_csv(traincsv)
    valdf = pd.read_csv(valcsv)

    train_dataset = CustlrDataset(traindf,config, mode='NOCROP',base_path = base_path)
    val_dataset = CustlrDataset(valdf,config, mode='NOCROP',base_path = base_path)
    out_channel = train_dataset[0]['landmark_map'].shape[0]
    print("Number of out channels " +str(out_channel))
    evaluator = _evaluator(config,out_channel)

    print('training set', len(train_dataset))
    print('val set', len(val_dataset))
    dssize = len(train_dataset)+len(val_dataset)

    trainloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
    valloader = DataLoader(val_dataset, batch_size=10, shuffle=True, num_workers=0)
    
    logname = time.strftime('%m-%d %H-%M-%S', time.localtime())
    logname = logname + ' param ' + str(0)+'-51' + 'dataset size '+ str(dssize)
    logname = 'runs/'+logname
    writer = SummaryWriter(logname)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    train_pos_loss = []
    val_pos_loss = []
    NUMSTEP = len(trainloader)
    step = 0
    for epoch in range(NUMEPOCH):
        for stat in ['train','val']:
            #print(stat)
            if stat == 'train':
                print("---Starting traing---")
                model.train()
                for i, sample in enumerate(trainloader):
                    step += 1
                    for key in sample:
                        sample[key] = sample[key].to("cuda")
                    output = model(sample)
                    #print(output['lm_pos_map'])

                    loss = model.cal_loss(sample, output)
                    optimizer.zero_grad()
                    loss['all'].backward()
                    optimizer.step()

                    if 'lm_vis_loss' in loss:
                        writer.add_scalar('loss/lm_vis_loss', loss['lm_vis_loss'], step)
                        writer.add_scalar('loss_weighted/lm_vis_loss', loss['weighted_lm_vis_loss'], step)
                    if 'lm_pos_loss' in loss:
                        writer.add_scalar('loss/lm_pos_loss', loss['lm_pos_loss'], step)
                        writer.add_scalar('loss_weighted/lm_pos_loss', loss['weighted_lm_pos_loss'], step)
                    writer.add_scalar('loss_weighted/all', loss['all'], step)
                    writer.add_scalar('global/learning_rate', learning_rate, step)

                    train_pos_loss.append(loss['lm_pos_loss'])
                    print('EPOCH:', epoch+1,'/',NUMEPOCH,'step:', i+1,'/',NUMSTEP,'| lm position loss:', loss['lm_pos_loss'],' | weighted loss:', loss['weighted_lm_pos_loss'])
                print("---End traing---")
            else:
                print("---Start evaluation---")
                model.eval()
                for i, sample in enumerate(tqdm(valloader)):
                    step += 1
                    for key in sample:
                        sample[key] = sample[key].to(device)
                    output = model(sample)
                    #print(output['lm_pos_map'])
                    evaluator.add(output, sample)
                ret = evaluator.evaluate()
                for i in range(len(config.CONSTANT.lm2name)):
                    print('metrics/dist_part_{}_{}'.format(i, config.CONSTANT.lm2name[i]), ret['lm_individual_dist'][i])
                    writer.add_scalar('metrics/dist_part_{}_{}'.format(i, config.CONSTANT.lm2name[i]), ret['lm_individual_dist'][i], step)
                print('metrics/dist_all', ret['lm_dist'])
                writer.add_scalar('metrics/dist_all', ret['lm_dist'], step)
                val_pos_loss.append(ret['lm_dist'])
                print('EPOCH:', epoch+1,'/',NUMEPOCH,'| lm distance:', ret['lm_dist'])
                model.train()
                print("---End evaluation---")
        LEARNING_RATE_DECAY =0.9
        learning_rate *= LEARNING_RATE_DECAY
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    try:
        model_file_name=config.MODEL.NAME+' '+new_model_name +' '+ str(0) + '-78'+' dataset '+str(dssize) +'.pkl'
        torch.save(model.state_dict(), './'+config.CONSTANT.SAVEFILE+'/' + model_file_name)
        print("MODEL SAVED:"+model_file_name)
    except Exception as e:
        print("AN ERROR HAS OCCURRED WHILE SAVING MODEL:"+str(e))
        
    
    plt.plot(train_pos_loss,label='training pos loss')
    
    tfigname = './'+config.CONSTANT.SAVEFILE+'/'+'experiment layers graph/training losslm/training loss '+new_model_name+' '+str(0)+' -78'+' dataset '+str(dssize)+' ' +config.MODEL.NAME+'.png'
    plt.savefig(tfigname)
    plt.show()
    
    plt.plot(val_pos_loss,label='validation pos loss')
    vfigname = './'+config.CONSTANT.SAVEFILE+'/'+ 'experiment layers graph/val losslm/val loss '+new_model_name +' '+str(0)+' -78'+' dataset '+str(dssize)+' ' +config.MODEL.NAME+'.png'
    plt.savefig(vfigname)
    plt.show()