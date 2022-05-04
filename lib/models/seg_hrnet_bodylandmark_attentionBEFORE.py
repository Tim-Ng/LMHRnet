# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import lm
from . import const

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace

BN_MOMENTUM = 0.1
ALIGN_CORNERS = None

logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=config.DATASET.NUM_CLASSES,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        
        x = self.stage4(x_list)
        hrnet_seg_x_list = x.copy()

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate (x[1], size=(x0_h, x0_w), mode='bilinear',align_corners=True )
        x2 = F.interpolate (x[2], size=(x0_h, x0_w), mode='bilinear',align_corners=True )
        x3 = F.interpolate (x[3], size=(x0_h, x0_w), mode='bilinear',align_corners=True )

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)

        return x,hrnet_seg_x_list

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            #for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model

# landmark branch creation 
class LandmarkBranch(nn.Module):

    def __init__(self, in_channel=256):
        super(LandmarkBranch, self).__init__()
        num_channels = lm.stage5_num_channels
        block = blocks_dict[lm.stage5_block]
        self.stage5, pre_stage_channels = self._make_stage(num_channels, multi_scale_output=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        self.last_layer_landmark = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                #number of landmarks
                out_channels=lm.stage_5_out_channels,
                kernel_size=lm.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if lm.FINAL_CONV_KERNEL == 3 else 0)
        )
    def _make_stage(self, num_inchannels,
                    multi_scale_output=True):
        num_modules = lm.stage5_Mods
        num_branches = lm.stage5_Num_branch
        num_blocks = lm.stage5_number_blocks
        num_channels = lm.stage5_num_channels
        block = blocks_dict[lm.stage5_block]
        fuse_method = lm.stage5_fuse_method

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self,GL_list):
        landmark = self.stage5(GL_list)
        # Upsampling
        landmark0_h, landmark0_w = landmark[0].size(2), landmark[0].size(3)
        landmark1 = F.interpolate (landmark[1], size=(landmark0_h, landmark0_w), mode='bilinear',align_corners=True )
        landmark2 = F.interpolate (landmark[2], size=(landmark0_h, landmark0_w), mode='bilinear',align_corners=True )
        landmark3 = F.interpolate (landmark[3], size=(landmark0_h, landmark0_w), mode='bilinear',align_corners=True )
        
        landmark = torch.cat([landmark[0], landmark1, landmark2, landmark3], 1)
        
        landmark = self.last_layer_landmark(landmark) 
        # lm_pos_map = F.sigmoid(x)
        lm_pos_map = landmark
        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, lm.stage_5_out_channels, -1)
        # y是高上的坐标，x是宽上的坐标
        argmax_out=torch.argmax(lm_pos_reshaped, dim=2).cpu()
        lm_pos_y, lm_pos_x = np.unravel_index(argmax_out, (pred_h, pred_w))
        lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)

        return lm_pos_map, lm_pos_output

class ModuleWithAttr(nn.Module):

    # 只能是数字，默认注册为0

    def __init__(self, extra_info=['step']):
        super(ModuleWithAttr, self).__init__()
        for key in extra_info:
            self.set_buffer(key, 0)

    def set_buffer(self, key, value):
        if not(hasattr(self, '__' + key)):
            self.register_buffer('__' + key, torch.tensor(value))
        setattr(self, '__' + key, torch.tensor(value))

    def get_buffer(self, key):
        if not(hasattr(self, '__' + key)):
            raise Exception('no such key!')
        return getattr(self, '__' + key).item()

class LandmarkExpNetwork(ModuleWithAttr):

    def __init__(self,cfg,**kwargs):
        super(LandmarkExpNetwork, self).__init__()
        #Load model to start
        self.seg_hrnet = get_seg_model(cfg,**kwargs)
        self.attention_pred_net1 = CustomUnetGenerator(48+20 , 48, num_downs=2, ngf=32, last_act='tanh')
        #self.attention_pred_net2 = CustomUnetGenerator(96+20 , 96, num_downs=2, ngf=32, last_act='tanh')
        #self.attention_pred_net3 = CustomUnetGenerator(192+20 , 192, num_downs=2, ngf=32, last_act='tanh')
        #self.attention_pred_net4 = CustomUnetGenerator(384+20 , 384, num_downs=2, ngf=32, last_act='tanh')
        self.LM_branch = LandmarkBranch()

    def forward(self, sample):
        batch_size, channel_num, image_h, image_w = sample['image'].size()
        hrnet_seg_x,hrnet_seg_x_list = self.seg_hrnet(sample['image'])
        
        attention_map1 = torch.cat([hrnet_seg_x_list[0], hrnet_seg_x], dim=1) #stage4_out[i] the output of stage 4, x = output
        attention_map1 = self.attention_pred_net1(attention_map1)
        
        #m2 = nn.MaxPool2d(28+1, stride=1)
        #hrnet_seg_x_28 = m2(hrnet_seg_x)
        #attention_map2 = torch.cat([hrnet_seg_x_list[1],hrnet_seg_x_28 ], dim=1) #stage4_out[i] the output of stage 4, x = output
        #print("attention_map2=")
        #print(attention_map2.size())
        #attention_map2 = self.attention_pred_net2(attention_map2)
        
        #m3 = nn.MaxPool2d(14+1, stride=1)
        #hrnet_seg_x_14 = m3(hrnet_seg_x_28)
        #attention_map3 = torch.cat([hrnet_seg_x_list[2], hrnet_seg_x_14], dim=1) #stage4_out[i] the output of stage 4, x = output
        #print("attention_map3=")
        #print(attention_map3.size())
        #attention_map3 = self.attention_pred_net3(attention_map3)
        
        #m4 = nn.MaxPool2d(7+1, stride=1)
        #hrnet_seg_x_7 = m4(hrnet_seg_x_14)
        #attention_map4 = torch.cat([hrnet_seg_x_list[3], hrnet_seg_x_7], dim=1) #stage4_out[i] the output of stage 4, x = output
        #print("attention_map4=")
        #print(attention_map4.size())
        #attention_map4 = self.attention_pred_net4(attention_map4)
        
        GL1 = (1 + attention_map1) * hrnet_seg_x_list[0]
        GL2 =  hrnet_seg_x_list[1] #*(1 + attention_map2) 
        GL3 = hrnet_seg_x_list[2]
        GL4 = hrnet_seg_x_list[3]
        GL_list = [GL1,GL2,GL3,GL4]
        lm_pos_map, lm_pos_output = self.LM_branch(GL_list)
        return {
            'lm_pos_output': lm_pos_output,
            'lm_pos_map': lm_pos_map,
            'segmentation_map' : hrnet_seg_x
        }

    def cal_loss(self, sample, output):
            batch_size, _, _, _ = sample['image'].size()
            lm_size = int(output['lm_pos_map'].shape[2])
            if hasattr(const, 'LM_TRAIN_USE') and const.LM_TRAIN_USE == 'in_pic':
                mask = sample['landmark_in_pic'].reshape(batch_size * lm.stage_5_out_channels, -1)
            else:
                mask = sample['landmark_vis'].reshape(batch_size * lm.stage_5_out_channels, -1)
            mask = torch.cat([mask] * lm_size * lm_size, dim=1).float()

            map_sample = sample['landmark_map%d' % lm_size].reshape(batch_size * lm.stage_5_out_channels, -1)
            map_output = output['lm_pos_map'].reshape(batch_size * lm.stage_5_out_channels, -1)
            #training loss
            lm_pos_loss = torch.pow(mask * (map_output - map_sample), 2).mean()

            all_loss = \
                lm.WEIGHT_LOSS_LM_POS * lm_pos_loss
            loss = {
                'all': all_loss,
                'lm_pos_loss': lm_pos_loss.item(),
                'weighted_lm_pos_loss': lm.WEIGHT_LOSS_LM_POS * lm_pos_loss.item(),
            }
            return loss
def get_seg_body_model_attention(cfg, **kwargs):
    model = LandmarkExpNetwork(cfg, **kwargs)

    return model