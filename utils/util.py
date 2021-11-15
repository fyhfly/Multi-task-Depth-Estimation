from __future__ import print_function
import torch
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from . import  evaluate

ignore_label = 255
id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                  3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                  7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                  14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                  18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                  28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1)
    image_numpy = image_numpy / (2.0 / 255.0)
    return image_numpy.astype(imtype)

def tensor2depth(input_depth, imtype=np.int32):
    if isinstance(input_depth, torch.Tensor):
        depth_tensor = input_depth.data
    else:
        return input_depth
    depth_numpy = depth_tensor[0].cpu().float().numpy() 
    depth_numpy += 1.0
    depth_numpy /= 2.0
    depth_numpy *= 65535.0
    depth_numpy = depth_numpy.reshape((depth_numpy.shape[1], depth_numpy.shape[2]))
    return depth_numpy.astype(imtype)

def tensor2segment(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        seg_tensor = input_image.data
    else:
        return input_image
    seg_numpy = seg_tensor[0].cpu().int().numpy()
    seg_numpy = np.squeeze(seg_numpy)
    return seg_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, imtype):
    image_pil = Image.fromarray(image_numpy, imtype)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

class SaveResults:
    def __init__(self, opt):
       
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.expr_name, 'image')
        mkdirs(self.img_dir) 
        self.log_name = os.path.join(opt.checkpoints_dir, opt.expr_name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def save_current_results(self, visuals, epoch):
            
        for label, image in visuals.items():
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            if image is None:
                continue
            if 'depth' in label:
                depth_numpy = tensor2depth(image)
                # save_image(depth_numpy, img_path, 'I')
                cmap = plt.cm.plasma
                norm = plt.Normalize(vmin=depth_numpy.min(), vmax=depth_numpy.max())
                heatmap = cmap(norm(depth_numpy))
                plt.imsave(img_path, heatmap)
            elif 'seg' in label:
                seg_numpy = tensor2segment(image)
                palette = evaluate.get_palette(256)
                segment = evaluate.id2trainId(seg_numpy, id_to_trainid, reverse=True)
                segment = Image.fromarray(segment.astype(np.uint8), 'L')
                segment.putpalette(palette)
                segment.save(img_path)
            else:
                image_numpy = tensor2im(image)
                save_image(image_numpy, img_path, 'RGB')
            

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, lr, losses, t, t_data):
          
        message = '(epoch: %d, iters: %d, lr: %e, time: %.3f, data: %.3f) ' % (epoch, i, lr, t, t_data)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
