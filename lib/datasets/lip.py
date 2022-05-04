# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from .base_dataset import BaseDataset


class LIP(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=20,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=473,
                 crop_size=(473, 473),
                 downsample_rate=1,
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(LIP, self).__init__(ignore_label, base_size,
                                  crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        
        self.label_mapping = {-1: ignore_label, 
                              0: ignore_label, 
                              1: ignore_label,
                              2: ignore_label, 
                              3: ignore_label,
                              4: ignore_label, 
                              5: ignore_label, 
                              6: ignore_label, 
                              7: 0, 
                              8: 1,
                              9: ignore_label, 
                              10: ignore_label,
                              11: 2,
                              12: 3, 
                              13: 4,
                              14: ignore_label, 
                              15: ignore_label, 
                              16: ignore_label,
                              17: 5,
                              18: ignore_label, 
                              19: 6,
                              20: 7,
                              21: 8,
                              22: 9,
                              23: 10,
                              24: 11,
                              25: 12,
                              26: 13,
                              27: 14,
                              28: 15, 
                              29: ignore_label,
                              30: ignore_label, 
                              31: 16, 
                              32: 17, 
                              33: 18}
        self.landmark_label_mapping = {
                              7: 0, 
                              17: 5,
                              22: 9,
                              27: 14,
                              28: 15, 
                              }
        
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            if 'train' in self.list_path:
                image_path, label_path, _ = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name, }
            elif 'val' in self.list_path:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name, }
            elif 'bodylandmark' in self.list_path:
                name = os.path.splitext(os.path.basename(item[0]))[0]
                sample = {"img": item[0],
                          "name": name, }
            else:
                raise NotImplementedError('Unknown subset.')
            files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        #item = self.files[index]
        #name = item["name"]
        #image_path = os.path.join(self.root, item['img'])
        #try:
        #    label_path = os.path.join(self.root, item['label'])
        #except:
        #    label = np.random.rand(3,2)
        #else:
        #    label = np.array(Image.open(label_path).convert('P'))

        #image = cv2.imread(
        #    image_path,
        #    cv2.IMREAD_COLOR
        #)

        #size = label.shape
        #if 'testval' in self.list_path:
        #    image = cv2.resize(image, self.crop_size,
        #                       interpolation=cv2.INTER_LINEAR)
        #    image = self.input_transform(image)
        #    image = image.transpose((2, 0, 1))

        #    return image.copy(), label.copy(), np.array(size), name

        #if self.flip:
        #    flip = np.random.choice(2) * 2 - 1
        #    image = image[:, ::flip, :]
        #    label = label[:, ::flip]

        #    if flip == -1:
        #        right_idx = [15, 17, 19]
        #        left_idx = [14, 16, 18]
        #        for i in range(0, 3):
        #            right_pos = np.where(label == right_idx[i])
        #            left_pos = np.where(label == left_idx[i])
        #            label[right_pos[0], right_pos[1]] = left_idx[i]
        #            label[left_pos[0], left_pos[1]] = right_idx[i]

        #image, label = self.resize_image(image, label, self.crop_size)
        #image, label = self.gen_sample(image, label,self.multi_scale, False)

        #return image.copy(), label.copy(), np.array(size), name
        item = self.files[index]
        name = item["name"]
        image_path = os.path.join(self.root, item['img'])
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))
        try:
            label_path = os.path.join(self.root, item['label'])
        except:
            label = np.random.rand(3,2)
        else:
            label = np.array(Image.open(label_path).convert('P'))
        convert_tensor = transforms.Compose([transforms.ToTensor()])
        tensor_img = convert_tensor(image)
        batch_t = torch.unsqueeze(tensor_img, 0)
        size = label.shape
        return image.copy(), label.copy(), np.array(size), name

    def inference(self, config, model, image, flip):
        size = image.size()
        pred = model(image)
        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))

            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]            

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            flip_output = flip_output.cpu()
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred[:, 14, :, :] = flip_output[:, 15, :, :]
            flip_pred[:, 15, :, :] = flip_output[:, 14, :, :]
            flip_pred[:, 16, :, :] = flip_output[:, 17, :, :]
            flip_pred[:, 17, :, :] = flip_output[:, 16, :, :]
            flip_pred[:, 18, :, :] = flip_output[:, 19, :, :]
            flip_pred[:, 19, :, :] = flip_output[:, 18, :, :]
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()
    
    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            print(str(i))
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
            seplist = self.sep_label(preds[i], inverse=True)
            print(pred)
            count= -1
            for j in seplist:
                save_sepimg = Image.fromarray(j)
                save_sepimg.putpalette(palette)
                save_sepimg.save(os.path.join(sv_path, name[i]+str(count)+'sep1.png'))
                count +=1
                
    #save method that seperates body, left hand, right hand, leg, and background into a list
    def sep_save_pred(self,preds,sv_path,name):
        total_sep_list = []
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            each_file = os.path.join(sv_path,name[i]+"sep")
            if not os.path.exists(each_file):
                os.mkdir(each_file)
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(each_file, name[i]+'.png'))
            seplist = self.sep_label(preds[i], inverse=True)
            count= -1
            total_sep_list.append(seplist)
            for j in seplist:
                save_sepimg = Image.fromarray(j)
                save_sepimg.putpalette(palette)
                save_sepimg.save(os.path.join(each_file, name[i]+str(count)+'sep1.png'))
                count +=1
        return total_sep_list
    def sep_label(self, label, inverse=False):
        temp = label.copy()
        seplist = []
        i=0;
        for v,k in self.landmark_label_mapping.items():
            oneFilter = label.copy()
            oneFilter[temp == v] = k
            oneFilter[temp != v] = 7
            seplist.append(oneFilter)
            i += 1
        return seplist;
        
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
