import collections
import math
import numbers
import random

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class RandomImgAugment(object):
    """Randomly shift gamma"""

    def __init__(self, size=None):

        self.size = size


    def __call__(self, inputs):

        img1 = inputs[0]
        img2 = inputs[1]
        depth = inputs[2]
        phase = inputs[3]
        fb = inputs[4]
        segment = inputs[5]

        h = img1.height
        w = img1.width
        w0 = w

        if self.size == [-1]:
            divisor = 32.0
            h = int(math.ceil(h/divisor) * divisor)
            w = int(math.ceil(w/divisor) * divisor)
            self.size = (h, w)
       
        scale_transform = transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])
        segment_scale = transforms.Resize(self.size)
        
        img1 = scale_transform(img1)
        if img2 is not None:
            img2 = scale_transform(img2)

        if fb is not None:
            scale = float(self.size[1]) / float(w0)
            fb = fb * scale
        if segment is not None:
            segment = segment_scale(segment)
        if phase == 'test':
            return img1, img2, depth, fb, segment
    
        if depth is not None:
           scale_transform_d = transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])
           depth = scale_transform_d(depth)

        if not self.size == 0:
    
            if depth is not None:
                arr_depth = np.array(depth, dtype=np.float32)
                arr_depth /= 65535.0  # cm->m, /10

                arr_depth[arr_depth<0.0] = 0.0
                depth = Image.fromarray(arr_depth, 'F')

        return img1, img2, depth, fb, segment

class SegToTensor(object):
    def __call__(self, input):
        arr_input = np.array(input).astype(np.float32)
        tensors = torch.from_numpy(arr_input.reshape((1, arr_input.shape[0], arr_input.shape[1])))
        return tensors

