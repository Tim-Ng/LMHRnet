import torch
import pandas as pd
import numpy as np
import importlib
import argparse

class LandmarkEvaluator(object):

    def __init__(self,cfg,out_channel=8):
        self.config =cfg
        self.out_channel = out_channel
        self.reset()

    def reset(self):
        self.lm_vis_count_all = np.array([0.] * self.out_channel) # 8 is the default number of output channels (heatmap), we need to change it to 13 or other number when necessary
        self.lm_dist_all = np.array([0.] * self.out_channel)

    def landmark_count(self, output, sample):
        #if hasattr(self.config.CONSTANT.LM_EVAL_USE, 'LM_EVAL_USE') and self.config.CONSTANT.LM_EVAL_USE == 'in_pic':
        #    mask_key = 'landmark_in_pic'
        #else:  
        mask_key = 'landmark_vis'
        landmark_vis_count = sample[mask_key].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample[mask_key].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        #validation loss
        landmark_dist = np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float * output['lm_pos_output'] - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
        ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def add(self, output, sample):
        self.landmark_count(output, sample)

    def evaluate(self):
        lm_individual_dist = self.lm_dist_all / self.lm_vis_count_all
        lm_dist = (self.lm_dist_all / self.lm_vis_count_all).mean()
        return {
            'category_accuracy_topk': {},
            'attr_group_recall': {},
            'attr_recall': {},
            'lm_individual_dist': lm_individual_dist,
            'lm_dist': lm_dist,
        }


#def merge_const(module_name):
##    new_conf = importlib.import_module(module_name)
#    for key, value in new_conf.__dict__.items():
#        if not(key.startswith('_')):
#            setattr(const, key, value)
#            print('override', key, value)


#def parse_args_and_merge_const():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--conf', default='', type=str)
##    args = parser.parse_args()
#    if args.conf != '':
#        merge_const(args.conf)
