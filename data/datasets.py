import collections
import glob
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils import data
from utils.dataset_util import KITTI
import random
import cv2

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):

        dd = {}
        {dd.update(d[i]) for d in self.datasets if d is not None}
        
        return dd

    def __len__(self):
        return max(len(d) for d in self.datasets if d is not None)


class KittiDataset(data.Dataset):
    def __init__(self, root='datasets', data_file='train.txt', phase='train',
                 img_transform=None, joint_transform=None, Seg_transform=None):
      
        self.root = root
        self.data_file = data_file
        self.files = []
        self.phase = phase
        self.img_transform = img_transform
        self.joint_transform = joint_transform
        self.Seg_transform = Seg_transform

        with open(osp.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                
                data_info = data.split(' ')

                self.files.append({
                    "l_rgb": data_info[0],
                    "r_rgb": data_info[1],
                    "cam_intrin": data_info[2],
                    "depth": data_info[3],
                    "segmentation": data_info[4]
                    })
                                    
    def __len__(self):
        return len(self.files)

    def read_data(self, datafiles):
        #print(osp.join(self.root, datafiles['l_rgb']))
        assert osp.exists(osp.join(self.root, datafiles['l_rgb'])), "Image does not exist"
        l_rgb = Image.open(osp.join(self.root, datafiles['l_rgb'])).convert('RGB')
        w = l_rgb.size[0]
        h = l_rgb.size[1]
        assert osp.exists(osp.join(self.root, datafiles['r_rgb'])), "Image does not exist"
        r_rgb = Image.open(osp.join(self.root, datafiles['r_rgb'])).convert('RGB')

        kitti = KITTI()
        assert osp.exists(osp.join(self.root, datafiles['cam_intrin'])), "Camera info does not exist"
        fb = kitti.get_fb(osp.join(self.root, datafiles['cam_intrin']))
        assert osp.exists(osp.join(self.root, datafiles['depth'])), "Depth does not exist"
        depth = kitti.get_depth(osp.join(self.root, datafiles['cam_intrin']),
                                osp.join(self.root, datafiles['depth']), [h, w])

        assert osp.exists(osp.join(self.root, datafiles['segmentation'])), "Segmentation does not exist"
        segmentation = Image.open(osp.join(self.root, datafiles['segmentation']))
        # segmentation = kitti.get_seman(osp.join(self.root, datafiles['segmentation']))

        return l_rgb, r_rgb, fb, depth, segmentation
    
    def __getitem__(self, index):
        if self.phase == 'train':
            index = random.randint(0, len(self)-1)
        if index > len(self)-1:
            index = index % len(self)
        datafiles = self.files[index]
        l_img, r_img, fb, depth, segmentation = self.read_data(datafiles)
            
        if self.joint_transform is not None:
            if self.phase == 'train':
                l_img, r_img, _, fb, segmentation = self.joint_transform((l_img, r_img, None, 'train', fb, segmentation))
            else:
                l_img, r_img, _, fb, segmentation = self.joint_transform((l_img, r_img, None, 'test', fb, segmentation))

        kitti = KITTI()
        segmentation = kitti.get_seman(segmentation)
            
        if self.img_transform is not None:
            l_img = self.img_transform(l_img)
            if r_img is not None:
                r_img = self.img_transform(r_img)

        if self.Seg_transform is not None:
            segmentation = self.Seg_transform(segmentation)

        if self.phase =='test':
            data = {}
            data['left_img'] = l_img
            data['depth'] = depth
            data['fb'] = fb
            data['segmentation'] = segmentation
            return data

        data = {}
        if l_img is not None:
            data['left_img'] = l_img
        if r_img is not None:
            data['right_img'] = r_img
        if fb is not None:
            data['fb'] = fb
        if segmentation is not None:
            data['segmentation'] = segmentation

        return data

# just for test 
class CityscapesDataset(data.Dataset):
    def __init__(self, root='./datasets', data_file='test_sceneflow.list', phase='test',
                 img_transform=None, joint_transform=None, Seg_transform=None):
        self.root = root
        self.data_file = data_file
        self.files = []
        self.phase = phase
        self.img_transform = img_transform
        self.joint_transform = joint_transform

        with open(osp.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                
                data_info = data.split(' ')

                self.files.append({
                    "rgb": data_info[0],
                    })
                                    
    def __len__(self):
        return len(self.files)

    def read_data(self, datafiles):
       
        # print(osp.join(self.root, datafiles['rgb']))
        assert osp.exists(osp.join(self.root, datafiles['rgb'])), "Image does not exist"
        rgb = Image.open(osp.join(self.root, datafiles['rgb'])).convert('RGB')
        
        disp = cv2.imread(osp.join(self.root, datafiles['rgb'].replace('image_2', 'disp_noc_0')), -1)
        disp = disp.astype(np.float32)/256.0
        return rgb, disp
    
    def __getitem__(self, index):
        index = index % len(self)
        datafiles = self.files[index]
        img, disp = self.read_data(datafiles)
            
        if self.joint_transform is not None:
                img, _, _, _, _ = self.joint_transform((img, None, None, 'test', None, None))
            
        if self.img_transform is not None:
            img = self.img_transform(img)

        data = {}
        data['left_img'] = img
        data['depth'] = disp
        return data

def get_dataset(root, data_file='train.list', dataset='kitti', phase='train',
                img_transform=None, Seg_transform=None,
                joint_transform=None):

    DEFINED_DATASET = {'KITTI', 'CITYSCAPES'}
    assert dataset.upper() in DEFINED_DATASET
    name2obj = {'KITTI': KittiDataset,
                'CITYSCAPES': CityscapesDataset,
        }

    return name2obj[dataset.upper()](root=root, data_file=data_file, phase=phase,
                                     img_transform=img_transform, Seg_transform=Seg_transform,
                                     joint_transform=joint_transform)

