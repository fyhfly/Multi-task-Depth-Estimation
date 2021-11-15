import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
from utils.bilinear_sampler import *
import torchvision.models as models
from torch.autograd import Variable
import math

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net

def init_mtl(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)
    return net

def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:,:,:-1] - img[:,:,:,1:]
    return gy

def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)

    return grid

def ssim(x, y):

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        C3 = C2 / 2
        
        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)

        sigma_x = F.avg_pool2d(x**2, 3, 1) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, 3, 1) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, 3, 1) - mu_x*mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d
        return torch.clamp((1-SSIM)/2, 0, 1)
    
##############################################################################
# Classes
##############################################################################

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def forward(self, depth, image):
        depth_grad_x = gradient_x(depth)
        depth_grad_y = gradient_y(depth)
        image_grad_x = gradient_x(image)
        image_grad_y = gradient_y(image)

        weights_x = torch.exp(-torch.mean(torch.abs(image_grad_x),1,True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_grad_y),1,True))
        smoothness_x = depth_grad_x*weights_x
        smoothness_y = depth_grad_y*weights_y

        loss_x = torch.mean(torch.abs(smoothness_x))
        loss_y = torch.mean(torch.abs(smoothness_y))


        loss = loss_x + loss_y
        
        return loss

class ReconLoss(nn.Module):
    def __init__(self, alpha=0.85):
        super(ReconLoss, self).__init__()
        self.alpha = alpha

    def forward(self, img0, img1, pred, fb, warp_path, max_d=655.35):

        x0 = (img0 + 1.0) / 2.0
        x1 = (img1 + 1.0) / 2.0

        assert x1.shape[0] == pred.shape[0]
        assert pred.shape[0] == fb.shape[0]

        new_depth = (pred + 1.0) / 2.0
        new_depth *= max_d
        disp = 1.0 / (new_depth+1e-6)
        tmp = np.array(fb)
        for i in range(new_depth.shape[0]):
            disp[i,:,:,:] *= tmp[i]
            disp[i,:,:,:] /= disp.shape[3] # normlize to [0,1]
        x0_w = bilinear_sampler_1d_h(x1, warp_path*disp)
        # x1_w = bilinear_sampler_1d_h(x0, -1*warp_path*disp)

        width = x1.shape[3]
        # width_l = int(width * 0.03594771) + 1
        # width_r = int(width * 0.96405229)

        # x0_crop = x0[:,:,:,width_l:width_r]
        # x0_w_crop = x0_w[:,:,:,width_l:width_r]

        ssim_ = ssim(x0, x0_w)
        l1 = torch.abs(x0-x0_w)
        # ssim_ = ssim(x1_crop, x1_w_crop)
        # l1 = torch.abs(x1_crop-x1_w_crop)

        loss1 = torch.mean(self.alpha * ssim_)
        loss2 = torch.mean((1-self.alpha) * l1)
        loss = loss1 + loss2

        recon_img = x0_w * 2.0-1.0
        # recon_img = x1_w * 2.0-1.0

        return loss, recon_img
#  CE loss
class CriterionDSN(nn.Module):

    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):

        target = torch.squeeze(target)
        loss1 = self.criterion(preds, target.long())

        return loss1

# Focal loss
class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num=19, gamma=2, ignore_index=255, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1)).cuda()
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(
            weight=self.alpha, reduction='none', ignore_index=ignore_index)

    def forward(self, predict, target):
        predict = predict.permute(0,2,3,1).reshape(-1, 19)
        target = target.long().view(-1)
        unignore_mask = target != self.ignore_index
        target = target[unignore_mask]
        predict = predict[unignore_mask]
        log_p = F.log_softmax(predict, dim=-1)
        ce = self.nll_loss(log_p, target)
        all_rows = torch.arange(len(predict))
        log_pt = log_p[all_rows, target]
        pt = log_pt.exp()
        focal_term = (1-pt)**self.gamma
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class _DecoderUpBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_DecoderUpBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.ConvTranspose2d(middle_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, use_bias=False):
        super(_OutputBlock, self).__init__()
        model = [
            nn.ReflectionPad2d(int(kernel_size/2)),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out

class _EncoderBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_EncoderBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.Conv2d(middle_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, input_nc=3, ngf=64, layers=4, norm='batch', weight=0.1):
        super(Encoder, self).__init__()

        self.layers = layers
        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type='PReLU')

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # encoder part
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nonlinearity
        )
        self.conv2 = _EncoderBlock(ngf, ngf*2, ngf*2, norm_layer, nonlinearity, use_bias)
        self.conv3 = _EncoderBlock(ngf*2, ngf*4, ngf*4, norm_layer, nonlinearity, use_bias)
        self.conv4 = _EncoderBlock(ngf*4, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)

        
    def forward(self, input):
        self.features = []
        conv1 = self.pool(self.conv1(input))
        self.features.append(conv1)
        conv2 = self.pool(self.conv2.forward(self.features[-1]))
        self.features.append(conv2)
        conv3 = self.pool(self.conv3.forward(self.features[-1]))
        self.features.append(conv3)
        center_in = self.pool(self.conv4.forward(self.features[-1]))
        self.features.append(center_in)

        return self.features

class Decoder(nn.Module):
    def __init__(self, ngf=64, dep_output_nc=1, seg_output_nc=19, layers=3, norm='batch', drop_rate=0, weight=0.1):
        super(Decoder, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type='PReLU')

        self.weight = weight
        self.layers = layers

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        for i in range(layers):
            layer = nn.Sequential(
                nn.Conv2d(ngf*8, ngf*8, kernel_size=3, padding=1, bias=use_bias),
                norm_layer(ngf*8),
                nonlinearity,
                nn.ReflectionPad2d(i*2+1),
                nn.Conv2d(ngf*8, ngf*8, kernel_size=3, padding=0, dilation=i*2+1, bias=use_bias)
            )
            setattr(self, 'layer'+str(i), layer)
        self.nonlinearity = nonlinearity
        self.norm1 = norm_layer(ngf*8)
        self.gate1 = nn.Linear(512*12*40, 3)
        self.gate2 = nn.Linear(512*12*40, 3)

        self.dep_deconv5 = _DecoderUpBlock(ngf*8, ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
        self.dep_deconv4 = _DecoderUpBlock(ngf*(4+4), ngf*8, ngf*2, norm_layer, nonlinearity, use_bias)
        self.dep_deconv3 = _DecoderUpBlock(ngf*(2+2), ngf*4, ngf, norm_layer, nonlinearity, use_bias)
        self.dep_deconv2 = _DecoderUpBlock(ngf*(1+1), ngf*2, int(ngf/2), norm_layer, nonlinearity, use_bias)

        self.dep_output1 = _OutputBlock(int(ngf/2), dep_output_nc, 7, use_bias)

        self.seg_deconv5 = _DecoderUpBlock(ngf*8, ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
        self.seg_deconv4 = _DecoderUpBlock(ngf*(4+4), ngf*8, ngf*2, norm_layer, nonlinearity, use_bias)
        self.seg_deconv3 = _DecoderUpBlock(ngf*(2+2), ngf*4, ngf, norm_layer, nonlinearity, use_bias)
        self.seg_deconv2 = _DecoderUpBlock(ngf*(1+1), ngf*2, int(ngf/2), norm_layer, nonlinearity, use_bias)

        self.seg_output1 = _OutputBlock(int(ngf/2), seg_output_nc, 7, use_bias)

    def forward(self, input_features):
        result = []
        x = input_features[-1]
        for i in range(self.layers):
            layer = getattr(self, 'layer'+str(i))
            result.append(layer(x))
        gate_x = x.view(x.size(0),-1)
        gate1 = F.softmax(self.gate1(gate_x),1)
        gate1 = gate1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gate2 = F.softmax(self.gate2(gate_x),1)
        gate2 = gate2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        center_dep = torch.mul(gate1[:,0,...],result[0]) + torch.mul(gate1[:,1,...],result[1]) + torch.mul(gate1[:,2,...],result[2])
        center_seg = torch.mul(gate2[:,0,...],result[0]) + torch.mul(gate2[:,1,...],result[1]) + torch.mul(gate2[:,2,...],result[2])
        center_dep = self.nonlinearity(self.norm1(center_dep)+x)
        center_seg = self.nonlinearity(self.norm1(center_seg)+x)

        dep_deconv5 = self.dep_deconv5.forward(center_dep)
        seg_deconv5 = self.seg_deconv5.forward(center_seg)

        dep_deconv4 = self.dep_deconv4.forward(torch.cat([dep_deconv5, input_features[2]*self.weight], 1))
        seg_deconv4 = self.seg_deconv4.forward(torch.cat([seg_deconv5, input_features[2]*self.weight], 1))

        dep_deconv3 = self.dep_deconv3.forward(torch.cat([dep_deconv4, input_features[1] * self.weight*0.5], 1))
        seg_deconv3 = self.seg_deconv3.forward(torch.cat([seg_deconv4, input_features[1] * self.weight*0.5], 1))

        dep_deconv2 = self.dep_deconv2.forward(torch.cat([dep_deconv3, input_features[0] * self.weight * 0.1], 1))
        seg_deconv2 = self.seg_deconv2.forward(torch.cat([seg_deconv3, input_features[0] * self.weight * 0.1], 1))

        depth = self.dep_output1.forward(dep_deconv2)
        segment = self.seg_output1.forward(seg_deconv2)

        return depth, segment

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, loss1, loss2):
        precision1 = torch.exp(-self.log_vars[0])
        loss = precision1 * loss1 ** 2. + self.log_vars[0]
        precision2 = torch.exp(-self.log_vars[1])
        loss += precision2 * math.log(loss2) + self.log_vars[1]

        return loss, self.log_vars.data.tolist()