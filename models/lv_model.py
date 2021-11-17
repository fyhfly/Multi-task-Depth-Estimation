import torch
import itertools
from .base_model import BaseModel
from . import lv_networks
from utils.image_pool import ImagePool
from utils import dataset_util
import math

class LVModel(BaseModel):
    def name(self):
        return 'LVModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_seg_con', type=float, default=10.0,
                                help='weight for warped segmentation consistency')
            parser.add_argument('--lambda_depth_recon', type=float, default=50.0,
                                help='weight for image reconstruction')
            parser.add_argument('--lambda_seg_recon', type=float, default=1,
                                help='weight for segmentation reconstruction')
            parser.add_argument('--lambda_Smooth', type=float, default=1, help='weight for smooth loss')
            parser.add_argument('--lambda_depth_con', type=float, default=1, help='weight for l f consistency')

            parser.add_argument('--Encoder_premodel', type=str, default="./checkpoints/kitti_lv/25_net_Encoder.pth",
                                help='pretrained Encoder model')
            parser.add_argument('--Decoder_premodel', type=str, default="./checkpoints/kitti_lv/25_net_Decoder.pth",
                                help='pretrained Decoder model')
            parser.add_argument('--mtl_premodel', type=str, default="./checkpoints/kitti_lv/25_net_mtl.pth",
                                help='pretrained mtl model')
            parser.add_argument('--freeze_bn', action='store_true', help='freeze the bn in mde')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        if self.isTrain:
            self.loss_names = ['depth_recon', 'flip_recon', 'seg_con', 'dep_con', 'f_dep_con']
            self.loss_names += ['smooth_l', 'smooth_f']
            # self.loss_names += ['seg_recon', 'lseg_recon']
            self.loss_names += ['delta1', 'delta2']

        if self.isTrain:
            self.visual_names = ['left_img', 'right_img', 'segment', 'l_segment', 'l_depth', 'warp_l_img']
            self.visual_names += ['flip_right_seg', 'f_segment', 'f_depth_flip', 'warp_f_img']
            # self.visual_names += ['warp_r_seg', 'warp_l_seg']

        else:
            self.visual_names = ['img', 'depth', 'segment']

        if self.isTrain:
            self.model_names = ['Encoder', 'Decoder']
            self.model_names += ['mtl']
        else:
            self.model_names = ['Encoder', 'Decoder']

        self.net_Encoder = lv_networks.init_net(lv_networks.Encoder(norm='batch'), init_type='kaiming', gpu_ids=opt.gpu_ids)
        self.net_Decoder = lv_networks.init_net(lv_networks.Decoder(norm='batch'), init_type='kaiming', gpu_ids=opt.gpu_ids)
        self.net_mtl = lv_networks.init_mtl(lv_networks.MultiTaskLoss(), gpu_ids=opt.gpu_ids)

        if self.isTrain:

            self.init_with_pretrained_model('Encoder', self.opt.Encoder_premodel)
            self.init_with_pretrained_model('Decoder', self.opt.Decoder_premodel)
            self.init_with_pretrained_model('mtl', self.opt.mtl_premodel)


        if self.isTrain:
            # define loss functions
            self.criterionImgRecon = lv_networks.ReconLoss()
            self.criterionCon = torch.nn.L1Loss()
            self.criterionCrossEntropy = lv_networks.CriterionDSN()
            self.criterionSmooth = lv_networks.SmoothLoss()
            self.criterionGAN = lv_networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)

            self.fake_seg_pool = ImagePool(opt.pool_size)
            self.fake_warp_pool = ImagePool(opt.pool_size)

            self.optimizer_G = torch.optim.AdamW([{'params': itertools.chain(self.net_Encoder.parameters(), self.net_Decoder.parameters())},
                                                {'params': self.net_mtl.parameters(), 'lr': 1e-3, 'eps': 1e-7}],
                                                lr=opt.lr, betas=(0.95, 0.99))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            if opt.freeze_bn:
                self.net_Decoder.apply(lv_networks.freeze_bn)

    def set_input(self, input):

        if self.isTrain:
            self.left_img = input['left_img'].to(self.device)
            self.right_img = input['right_img'].to(self.device)
            self.fb = input['fb']
            self.segment = input['segmentation'].to(self.device)
            self.flip_right_seg = torch.flip(self.segment, [-1])
            self.flip_left_img = torch.flip(self.right_img, [-1])
            self.flip_right_img = torch.flip(self.left_img, [-1])
        else:
            self.img = input['left_img'].to(self.device)

    def forward(self):

        if self.isTrain:
            pass

        else:
            self.feature_map = self.net_Encoder(self.img)
            self.d, self.s = self.net_Decoder(self.feature_map)
            # self.flip = torch.flip(self.img, [-1])
            # self.f, self.fs = self.net_Decoder(self.net_Encoder(self.flip))
            # self.fs = torch.flip(self.fs, [-1])
            # self.s = (self.s+self.fs)*0.5
            self.segment = self.s.argmax(dim=1, keepdim=True).float()
            # self.d = torch.unsqueeze(self.d, 0)
            # self.f = torch.unsqueeze(self.f, 0)
            # self.d =  torch.cat((self.d, self.f), 0)
            self.depth = self.d


    def backward_G(self):

        lambda_depth_recon = self.opt.lambda_depth_recon
        lambda_seg_recon = self.opt.lambda_seg_recon
        lambda_seg_con = self.opt.lambda_seg_con
        lambda_Smooth = self.opt.lambda_Smooth
        lambda_depth_con = self.opt.lambda_depth_con

        self.l_feature_map = self.net_Encoder(self.left_img)
        self.l_d, self.l_s = self.net_Decoder(self.l_feature_map)
        self.l_segment = self.l_s.argmax(dim=1, keepdim=True).float()
        self.l_depth = self.l_d

        self.f_feature_map = self.net_Encoder(self.flip_left_img)
        self.f_d, self.f_s = self.net_Decoder(self.f_feature_map)
        self.f_segment = self.f_s.argmax(dim=1, keepdim=True).float()
        self.f_depth = self.f_d
        self.f_depth_flip = torch.flip(self.f_depth, [-1])

        self.r_segment = torch.flip(self.f_segment, [-1])

        # left warp
        self.loss_depth_recon, self.warp_l_img =self.criterionImgRecon(self.left_img, self.right_img, self.l_depth, self.fb, warp_path=-1.0)
        self.loss_depth_recon *= lambda_depth_recon
        # left smooth
        self.loss_smooth_l = self.criterionSmooth(self.l_depth, self.left_img) * lambda_Smooth
        self.loss_smooth_l += self.criterionSmooth(self.l_depth, self.segment) * lambda_Smooth

        # right warp
        self.loss_flip_recon, self.warp_f_img = self.criterionImgRecon(self.flip_left_img, self.flip_right_img, self.f_depth, self.fb, warp_path=-1.0)
        self.loss_flip_recon *= lambda_depth_recon
        self.warp_f_img = torch.flip(self.warp_f_img, [-1])
        # right smooth
        self.loss_smooth_f = self.criterionSmooth(self.f_depth, self.flip_left_img) * lambda_Smooth
        self.loss_smooth_f += self.criterionSmooth(self.f_depth, self.f_segment) * lambda_Smooth

        # # flip semantic warp
        # self.loss_seg_recon, self.warp_r_seg = self.criterionImgRecon(self.f_segment, self.flip_right_seg, self.f_depth, self.fb, warp_path=-1.0)
        # self.loss_seg_recon *= lambda_seg_recon
        # # semantic warp
        # self.loss_lseg_recon, self.warp_l_seg = self.criterionImgRecon(self.segment, self.r_segment, self.l_depth, self.fb, warp_path=-1.0)
        # self.loss_lseg_recon *= lambda_seg_recon

        self.loss_dep_con, _ = self.criterionImgRecon(self.l_depth, self.f_depth_flip, self.l_depth, self.fb, warp_path=-1.0)
        self.loss_dep_con  *= lambda_depth_con
        f_r_depth = torch.flip(self.l_depth, [-1])
        self.loss_f_dep_con, _ = self.criterionImgRecon(self.f_depth, f_r_depth, self.f_depth, self.fb, warp_path=-1.0)
        self.loss_f_dep_con *=  lambda_depth_con

        # left semantic
        self.loss_seg_con = self.criterionCrossEntropy(self.l_s, self.segment) * lambda_seg_con

        self.loss_G, self.log_vars = self.net_mtl(self.loss_depth_recon + self.loss_flip_recon, self.loss_seg_con)
        self.loss_delta1 = math.exp(self.log_vars[0]) ** 0.5
        self.loss_delta2 = math.exp(self.log_vars[1]) ** 0.5
        self.loss_G += self.loss_dep_con + self.loss_f_dep_con + self.loss_smooth_l + self.loss_smooth_f
        self.loss_G.backward()

    def optimize_parameters(self):

        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

