"""
This script is used to define the class for LFZSSR including hyperparameters, loss functions, training and testing.
"""
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tools.utils import *
from tools.common import *
from tools.logger import make_logs
from configs.Config_for_LFZSSR import Config
from tensorboardX import SummaryWriter
from tools.dataloader_for_LFZSSR import *
from models.model_single_target_view import *


def get_cur_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def load_params(src_dict, des_model):
    # src_dict: state dict of source model
    # des_model: model of the destination model
    des_dict = des_model.state_dict()
    for_des = {k: v for k, v in src_dict.items() if k in des_dict.keys()}
    des_dict.update(for_des)
    des_model.load_state_dict(des_dict)
    return des_model

class LFZSSR_SingleTargetView:
    # define the counters
    t_iter = 0 # iter used for every iterative step
    # for AlignNet
    align_iter = 0
    # for FusionNet
    aggre_iter = 0
    aggre_test_PSNR = []
    aggre_test_iters = []
    # for finetune
    ft_iter = 0
    ft_test_PSNRs = []
    ft_test_iters = []
    # for VDSR
    vdsr_PSNR = 0
    vdsr_ref = 0

    def __init__(self, lf_name, mat_path, conf=Config(), save_name="", save_prefix="", repeat_iter=0, gpu_num=2):
        # define configuration parameters
        self.conf = conf
        self.save_name = "{}_{}".format(save_name, repeat_iter)
        self.save_prefix = save_prefix
        self.lf_name = lf_name
        self.resize = bicubic_imresize()
        self.gpu_num = gpu_num

        # define configurations from Config()
        self.lr_align_stage = self.conf.lr_align_stage
        self.lr_aggre_stage = self.conf.lr_aggre_stage
        self.lr_ft_stage = self.conf.lr_ft_stage
        self.scale = self.conf.scale
        self.patch_size = self.conf.patch_size
        self.view_num = self.conf.view_num
        self.refPos = self.conf.refPos
        self.level_num = self.conf.level_num

        # Set GPU device
        if self.conf.use_cuda:
            print("===> Using GPU with id: '{}'".format(self.conf.gpu_id))
            os.environ["CUDA_VISIBLE_DEVICES"] = self.conf.gpu_id
            if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong GPU id, please check gpu is or run without GPU")

        # Set dataloader
        self.dataloader_align = DataloaderForAlignNetWithBatch(mat_path=mat_path,
                                                               refPos=self.refPos,
                                                               scale=self.scale,
                                                               view_num=self.view_num,
                                                               patch_size=self.conf.align_patch_size,
                                                               batch_size=self.conf.align_batch_size)
        self.dataloader_aggre = DataloaderForLFZSSRWithBatch(mat_path=mat_path,
                                                             refPos=self.refPos,
                                                             scale=self.scale,
                                                             view_num=self.view_num,
                                                             patch_size=self.patch_size,
                                                             batch_size=self.conf.aggre_batch_size,
                                                             scale_aug=self.conf.scale_aug)
        self.dataloader_ft = DataloaderForLFZSSRWithBatch(mat_path=mat_path,
                                                       refPos=self.refPos,
                                                       scale=self.scale,
                                                       view_num=self.view_num,
                                                       patch_size=self.patch_size,
                                                       batch_size=self.conf.ft_batch_size,
                                                       scale_aug=False)
        # Record
        if self.conf.record:
            # writer
            self.writer = SummaryWriter(log_dir="{}/logs/{}/".format(self.save_prefix, self.save_name),
                                        comment="Training curve with {}".format(self.save_name))
            make_logs("{}/log/{}/".format(self.save_prefix, self.save_name), 'log.txt', 'err.txt')

        # Set the random seeds
        if not self.conf.random_seed:
            self.seed = random.randint(1, 10000)
        else:
            self.seed = self.conf.random_seed

        # Set the networks
        self.align_net = AlignNet(refPos=self.refPos, scale=self.scale,
                                  level_num=self.level_num,
                                  level_step=2.0 * self.conf.disp_max / (self.level_num - 1),
                                  pad_size=self.conf.pad_size,
                                  view_num=self.conf.view_num)
        self.align_net_test = AlignNet_ForTest(refPos=self.refPos, scale=self.scale,
                                               level_num=self.level_num,
                                               level_step=2.0 * self.conf.disp_max / (self.level_num - 1),
                                               pad_size=self.conf.pad_size,
                                               view_num=self.conf.view_num)
        self.align_aggre_net = AlignWithAggreNet(refPos=self.refPos, scale=self.scale,
                                                 level_num=self.level_num,
                                                 level_step=2.0 * self.conf.disp_max / (self.level_num - 1),
                                                 pad_size=self.conf.pad_size,
                                                 view_num=self.conf.view_num)
        self.align_aggre_net_test = AlignWithAggreNet_ForTest(refPos=self.refPos, scale=self.scale,
                                                              level_num=self.level_num,
                                                              level_step=2.0 * self.conf.disp_max / (self.level_num - 1),
                                                              pad_size=self.conf.pad_size,
                                                              view_num=self.conf.view_num)

        self.VDSR_model = torch.load(self.conf.vdsr_model_path)

        self.align_net = load_params(self.VDSR_model, self.align_net)
        self.align_aggre_net = load_params(self.VDSR_model, self.align_aggre_net)

        # Set loss criterions
        self.L1criterion = nn.L1Loss(reduction="mean")
        self.L2criterion = nn.MSELoss(reduction="mean")
        self.L2criterion_sum = nn.MSELoss(reduction="sum")

        # Feed models and loss functions into CUDA
        if self.conf.use_cuda:
            self.L1criterion = self.L1criterion.cuda()
            self.L2criterion = self.L2criterion.cuda()
            self.L2criterion_sum = self.L2criterion_sum.cuda()
            self.align_net = self.align_net.cuda()
            self.align_net_test = self.align_net_test.cuda()
            self.align_aggre_net = self.align_aggre_net.cuda()
            self.align_aggre_net_test = self.align_aggre_net_test.cuda()

        # Three optimizers for three stages
        self.align_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.align_net.parameters()),
                                          lr=self.lr_align_stage, weight_decay=self.conf.weight_decay)

        self.aggre_optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                                 self.align_aggre_net.parameters()),
                                          lr=self.lr_aggre_stage,
                                          weight_decay=self.conf.weight_decay_aggre)

        self.ft_optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                              self.align_aggre_net.parameters()),
                                       lr=self.lr_ft_stage,
                                       weight_decay=self.conf.weight_decay)


    def back_projection_loss(self, sr_ref, lr_lf):
        # sr_cv: [B, 1, sH, sW]
        # lr_lf: [B, UV, H, W]
        B, H, W = lr_lf.shape[0], lr_lf.shape[2], lr_lf.shape[3]
        lr_lf = lr_lf.view(B, -1, self.view_num, H, W)
        lr_ref = lr_lf[:, self.refPos[0], self.refPos[1], :, :].unsqueeze(1)
        recon_lr_ref = self.resize(sr_ref, 1.0/self.scale)
        return self.L1criterion(lr_ref, recon_lr_ref)

    def align_loss(self, warped_ref, gt_ref):
        gt_ref = gt_ref.repeat(1, warped_ref.shape[1], 1, 1)
        warp_loss = self.L2criterion_sum(gt_ref, warped_ref) / \
                    (warped_ref.shape[0] * warped_ref.shape[1] * warped_ref.shape[2] * warped_ref.shape[3])
        return warp_loss

    def early_stop(self, lr):
        # if lr is smaller than the min_lr, return an early stop signal
        if lr <= self.conf.min_learning_rate:
            return True
        else:
            return False

    #### ---- Define test for AlignNet

    def test_AlignNet(self):
        self.align_net.eval()

        # get the lr_lf
        lr_lf = self.dataloader_aggre.lf_lr
        hr_lf = self.dataloader_aggre.lf_hr
        hr_lf = lf_modcrop(hr_lf, self.scale)
        hr_ref_view = hr_lf[self.refPos[0], self.refPos[1]]  # [H, W]
        U = lr_lf.shape[0]
        # get the central light fields
        view_start = (U - self.view_num) // 2
        lr_lf = lr_lf[view_start: view_start+self.view_num, view_start: view_start+self.view_num]
        lr_lf = lr_lf.astype(float) / 255.0
        lr_lf = torch.Tensor(lr_lf).float().view(1, -1, lr_lf.shape[2], lr_lf.shape[3])
        # feed into devices
        if self.conf.use_cuda:
            lr_lf = lr_lf.cuda()
        else:
            lr_lf = lr_lf.cpu()
        # copy weights to test network
        trained_dict = self.align_net.module.state_dict()
        test_dict = self.align_net_test.state_dict()
        for_test = {k: v for k, v in trained_dict.items() if k in test_dict.keys()}
        test_dict.update(for_test)
        self.align_net_test.load_state_dict(test_dict)
        # set the aug_scale and test
        aug_scale = self.scale
        with torch.no_grad():
            try:
                warped_vdsr_lf, vdsr_cv, _, _ = self.align_net_test(lr_lf, aug_scale, self.conf.set_name)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        warped_vdsr_lf, vdsr_cv, _, _ = self.align_net_test(lr_lf, aug_scale, self.conf.set_name)
                else:
                    raise exception
        # calculate the warp mse of LR resolution (not used in training)
        vdsr_cv = vdsr_cv.repeat(1, warped_vdsr_lf.shape[1], 1, 1)
        test_mse = self.L2criterion(vdsr_cv, warped_vdsr_lf)
        test_mse = test_mse.cpu().data.numpy()
        print("WarpNet test mse: {:.10f}".format(test_mse))
        # record
        if self.conf.record:
            self.writer.add_scalar("scalar_align_stage/Test_mse", test_mse, self.align_iter)
        if self.align_iter == 0:
            vdsr_cv = vdsr_cv[:, 0, :, :].squeeze()
            vdsr_cv = vdsr_cv.cpu().data.numpy()
            vdsr_cv = transfer_img_to_uint8(vdsr_cv)
            self.vdsr_ref = vdsr_cv
            H, W = hr_ref_view.shape
            H = H - H % self.scale
            W = W - W % self.scale
            hr_ref_view = hr_ref_view[:H, :W]
            self.vdsr_PSNR = PSNR(self.vdsr_ref, hr_ref_view)
            print("VDSR PSNR is :{:.4f}".format(self.vdsr_PSNR))
        return test_mse

    def test(self, mode):
        self.align_aggre_net.eval()

        lr_lf = self.dataloader_aggre.lf_lr
        hr_lf = self.dataloader_aggre.lf_hr
        hr_lf = lf_modcrop(hr_lf, self.scale)
        U = lr_lf.shape[0]
        # get the central light fields
        view_start = (U - self.view_num) // 2
        lr_lf = lr_lf[view_start: view_start + self.view_num, view_start: view_start + self.view_num]
        hr_lf = hr_lf[view_start: view_start + self.view_num, view_start: view_start + self.view_num]
        hr_ref_view = hr_lf[self.refPos[0], self.refPos[1]] # [H, W]
        # transfer lr_lf into tensor
        lr_lf = lr_lf.astype(float) / 255.0
        lr_lf = torch.Tensor(lr_lf).float().view(1, -1, lr_lf.shape[2], lr_lf.shape[3])
        # feed into devices
        if self.conf.use_cuda:
            lr_lf = lr_lf.cuda()
        else:
            lr_lf = lr_lf.cpu()
        # set scale_aug and test
        aug_scale = self.scale
        ### --- Copy weights
        trained_dict = self.align_aggre_net.module.state_dict()
        test_dict = self.align_aggre_net_test.state_dict()
        for_test = {k: v for k, v in trained_dict.items() if k in test_dict.keys()}
        test_dict.update(for_test)
        self.align_aggre_net_test.load_state_dict(test_dict)
        ### test
        with torch.no_grad():
            try:
                _, _, sr_res, _, _ = self.align_aggre_net_test(lr_lf, aug_scale, self.conf.set_name)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        _, _, sr_res, _, _ = self.align_aggre_net_test(lr_lf, aug_scale, self.conf.set_name) # [1, 1, H, W]
                else:
                    raise exception
        # calculate the average PSNR
        sr_res = sr_res.cpu().data  # 1 1 H W
        sr_res = sr_res.squeeze()
        sr_res = sr_res.numpy().astype(np.float32)

        sr_res = transfer_img_to_uint8(sr_res)
        PSNR_value = PSNR(sr_res, hr_ref_view)

        # display
        print("Testing PSNR is: {:.4f}".format(PSNR_value))

        if self.conf.record:
            if mode == "fuse":
                self.writer.add_scalar("scalar_aggre_stage/Test_PSNR", PSNR_value, self.aggre_iter)
            elif mode == "ft":
                self.writer.add_scalar("scalar_ft_stage/Test_PSNR", PSNR_value, self.ft_iter)
            else:
                raise Exception("Wrong training mode!")

        return PSNR_value, sr_res

    #### ---- Training with three stages

    def trainer_align_stage(self):

        cudnn.benchmark = True
        # freeze VDSR parameters
        self.align_net.set_zero_lr_VDSR()
        self.align_net = nn.DataParallel(self.align_net)

        # train
        for self.align_iter in range(1, self.conf.max_iters):
            self.align_net.train()
            # change the learning rate if it's necessary
            for param_group in self.align_optimizer.param_groups:
                param_group["lr"] = self.lr_align_stage

            # get patch and start to train
            lf_patch = self.dataloader_align.get_patch()
            aug_scale = self.scale
            if self.conf.use_cuda:
                lf_patch = lf_patch.cuda()

            # forward
            self.align_optimizer.zero_grad()
            patch_h, patch_w = lf_patch.shape[2], lf_patch.shape[3]

            range_angular = torch.arange(0, self.view_num).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_x_lr_pad = torch.arange(0, patch_w + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_y_lr_pad = torch.arange(0, patch_h + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_x_hr_pad = torch.arange(0, patch_w * self.scale + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_y_hr_pad = torch.arange(0, patch_h * self.scale + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)


            aligned_vdsr_lf, vdsr_cv, lr_disp, bic_disp = self.align_net(lf_patch, aug_scale,
                                                                        range_angular,
                                                                        range_spatial_x_lr_pad,
                                                                        range_spatial_y_lr_pad,
                                                                        range_spatial_x_hr_pad,
                                                                        range_spatial_y_hr_pad)

            # backward
            align_loss = self.align_loss(aligned_vdsr_lf, vdsr_cv)
            align_loss.backward()
            self.align_optimizer.step()

            # record
            if self.conf.record:
                self.writer.add_scalar("scalar_align_stage/align_loss", align_loss.cpu().data, self.align_iter)
            # display
            if (self.align_iter % self.conf.display_loss_step == 0):
                print("===> {}: Iteration: {}, lr: {:.5f}, Warp loss: {:.10f}".
                      format(get_cur_time(), self.align_iter,
                             self.align_optimizer.param_groups[0]["lr"],
                             align_loss.cpu().data))
           
            if self.align_iter % 8000 == 0 and self.align_iter < 16000:
                self.lr_align_stage *= 0.1
                print("Learning rate of ALIGN stage is updated, now it's {:.10f}".format(self.lr_align_stage))

            if self.align_iter == (self.conf.max_iters - 1):
                print("We've got the maximum iteration point")
                return

    def trainer_aggre_stage(self):
        # This trainer is used for the second stage, we should load the pretrained AlignNet and freeze its parameters
        cudnn.benchmark = True
        # freeze VDSR
        self.align_aggre_net.set_zero_lr_VDSR()

        # load AlignNet
        align_dict = self.align_net.module.state_dict()
        align_aggre_dict = self.align_aggre_net.state_dict()
        dict_from_align_net = {k: v for k, v in align_dict.items() if k in align_aggre_dict.keys()}
        align_aggre_dict.update(dict_from_align_net)
        self.align_aggre_net.load_state_dict(align_aggre_dict)
        # freeze AlignNet
        self.align_aggre_net.set_zero_lr_AlignNet()
        self.align_aggre_net = nn.DataParallel(self.align_aggre_net)

        # train
        for self.aggre_iter in range(1, self.conf.max_iters):
            self.align_aggre_net.train()

            # change the learning rate if it's necessary
            # for param_group in self.ft_optimizer.param_groups:
            #     param_group["lr"] = self.lr_aggre_stage

            # get patches
            lr_patch, hr_patch, _, hr_ref_patch, aug_scale = self.dataloader_aggre.get_patch()
            if self.conf.use_cuda:
                lr_patch = lr_patch.cuda()
                hr_patch = hr_patch.cuda()
                hr_ref_patch = hr_ref_patch.cuda()
            # forward
            self.aggre_optimizer.zero_grad()
            patch_h, patch_w = hr_patch.shape[2], hr_patch.shape[3]

            range_angular = torch.arange(0, self.view_num).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_x_lr_pad = torch.arange(0, patch_w // self.scale + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_y_lr_pad = torch.arange(0, patch_h // self.scale + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_x_hr_pad = torch.arange(0, patch_w + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_y_hr_pad = torch.arange(0, patch_h + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)

            _, _, ref_sr_res, lr_disp, bic_disp = self.align_aggre_net(lr_patch, aug_scale,
                                                                       range_angular,
                                                                       range_spatial_x_lr_pad,
                                                                       range_spatial_y_lr_pad,
                                                                       range_spatial_x_hr_pad,
                                                                       range_spatial_y_hr_pad
                                                                       )
            if self.conf.zssr_bp_ratio > 0.0:
                range_spatial_x_hr_pad = torch.arange(0, patch_w + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
                range_spatial_y_hr_pad = torch.arange(0, patch_h + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
                range_spatial_x_hhr_pad = torch.arange(0, patch_w * self.scale + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
                range_spatial_y_hhr_pad = torch.arange(0, patch_h * self.scale + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
                aug_scale_hr = aug_scale * self.scale
                _, _, ref_hr_sr_res, _, _ = self.align_aggre_net(hr_patch, aug_scale_hr,
                                                                 range_angular,
                                                                 range_spatial_x_hr_pad,
                                                                 range_spatial_y_hr_pad,
                                                                 range_spatial_x_hhr_pad,
                                                                 range_spatial_y_hhr_pad)
                bp_loss = self.back_projection_loss(ref_hr_sr_res, hr_patch)
            # backward
            sr_loss = self.L1criterion(ref_sr_res, hr_ref_patch)
            if self.conf.zssr_bp_ratio > 0.0:
                total_loss = sr_loss + self.conf.zssr_bp_ratio * bp_loss
                total_loss.backward()
            else:
                sr_loss.backward()

            self.aggre_optimizer.step()

            # record
            if self.conf.record:
                self.writer.add_scalar("scalar_aggre_stage/SR_loss", sr_loss.cpu().data, self.aggre_iter)
                if self.conf.zssr_bp_ratio > 0.0:
                    self.writer.add_scalar("scalar_fuse_stage/BP_loss", bp_loss.cpu().data, self.aggre_iter)
                    self.writer.add_scalar("scalar_fuse_stage/Total_loss", total_loss.cpu().data, self.aggre_iter)
            # display
            if (self.aggre_iter % self.conf.display_loss_step == 0):
                if self.conf.zssr_bp_ratio > 0.0:
                    print("===> {}: Iteration: {}, lr: {:.5f}, SR loss: {:.10f}, "
                          "BP loss: {:.10f}, Total loss: {:.10f}".
                          format(get_cur_time(), self.aggre_iter,
                                 self.aggre_optimizer.param_groups[0]["lr"],
                                 sr_loss.cpu().data, bp_loss.cpu().data,
                                 total_loss.cpu().data))
                else:
                    print("===> {}: Iteration: {}, lr: {:.5f}, SR loss: {:.10f}".
                          format(get_cur_time(), self.aggre_iter,
                                 self.aggre_optimizer.param_groups[0]["lr"],
                                 sr_loss.cpu().data))

            # if self.aggre_iter == self.conf.warp_fusion_iter_step1:
            #     self.lr_aggre_stage *= 0.1
            #     print("Learning rate of AGGRE stage is updated, now it's {:.10f}".format(self.lr_aggre_stage))
            if self.aggre_iter == (self.conf.max_iter_aggre - 1):
                print("Get maximum iteration number, start testing!")
                PSNR_value, sr_res = self.test(mode="fuse")
                self.aggre_test_PSNR.append(PSNR_value)
                self.aggre_test_iters.append(self.aggre_iter // self.conf.test_step)
                return sr_res

    def ft_trainer(self):
        # this trainer is used for the third stage

        cudnn.benchmark = True
        print("Free WarpNet ------")
        self.align_aggre_net.module.free_AlignNet()

        # train
        for self.ft_iter in range(1, self.conf.max_iters):
            self.align_aggre_net.train()

            # change the learning rate if it's necessary
            # for param_group in self.ft_optimizer.param_groups:
            #     param_group["lr"] = self.lr_ft_stage

            # get patches
            lr_patch, hr_patch, _, hr_ref_patch, aug_scale = self.dataloader_ft.get_patch()
            if self.conf.use_cuda:
                lr_patch = lr_patch.cuda()
                hr_patch = hr_patch.cuda()
                hr_ref_patch = hr_ref_patch.cuda()
            # forward
            self.ft_optimizer.zero_grad()

            patch_h, patch_w = hr_patch.shape[2], hr_patch.shape[3]

            range_angular = torch.arange(0, self.view_num).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_x_lr_pad = torch.arange(0, patch_w // self.scale + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_y_lr_pad = torch.arange(0, patch_h // self.scale + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_x_hr_pad = torch.arange(0, patch_w + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
            range_spatial_y_hr_pad = torch.arange(0, patch_h + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)

            warped_vdsr_lf, vdsr_ref, ref_sr_res, lr_disp, bic_disp = self.align_aggre_net(lr_patch, aug_scale,
                                                                                           range_angular,
                                                                                           range_spatial_x_lr_pad,
                                                                                           range_spatial_y_lr_pad,
                                                                                           range_spatial_x_hr_pad,
                                                                                           range_spatial_y_hr_pad
                                                                                           )
            # vdsr_ref = vdsr_ref.repeat(1, warped_vdsr_lf.shape[1], 1, 1)
            # backward
            # for WarpNet
            align_lr_loss = self.align_loss(warped_vdsr_lf, vdsr_ref)

            # for FusionNet
            if self.conf.zssr_bp_ratio > 0.0:
                range_spatial_x_hr_pad = torch.arange(0, patch_w + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
                range_spatial_y_hr_pad = torch.arange(0, patch_h + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
                range_spatial_x_hhr_pad = torch.arange(0, patch_w * self.scale + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
                range_spatial_y_hhr_pad = torch.arange(0, patch_h * self.scale + 2 * self.conf.pad_size).view(1, -1).repeat(self.gpu_num, 1)
                aug_scale_hr = aug_scale * self.scale
                _, _, ref_hr_sr_res, _, _ = self.align_aggre_net(hr_patch, aug_scale_hr,
                                                                 range_angular,
                                                                 range_spatial_x_hr_pad,
                                                                 range_spatial_y_hr_pad,
                                                                 range_spatial_x_hhr_pad,
                                                                 range_spatial_y_hhr_pad)
                bp_loss = self.back_projection_loss(ref_hr_sr_res, hr_patch)
            else:
                bp_loss = 0.0
            sr_loss = self.L1criterion(ref_sr_res, hr_ref_patch)
            total_loss = sr_loss + align_lr_loss * self.conf.align_loss_weight + bp_loss * self.conf.zssr_bp_ratio
            total_loss.backward()
            self.ft_optimizer.step()

            # record
            if self.conf.record:
                self.writer.add_scalar("scalar_ft_stage/SR_loss", sr_loss.cpu().data, self.ft_iter)
                self.writer.add_scalar("scalar_ft_stage/Align_lr_loss", align_lr_loss.cpu().data, self.ft_iter)
                if self.conf.zssr_bp_ratio > 0.0:
                    self.writer.add_scalar("scalar_ft_stage/BP_loss", bp_loss.cpu().data, self.ft_iter)
                self.writer.add_scalar("scalar_ft_stage/Total_loss", total_loss.cpu().data, self.ft_iter)
            # display
            if (self.ft_iter % self.conf.display_loss_step == 0):
                if self.conf.zssr_bp_ratio > 0.0:
                    print(
                        "===> {}: Iteration: {}, lr: {:.6f}, SR loss: {:.10f}, BP loss: {:.10f}, "
                        "Align LR loss: {:.10f}, Total loss: {:.10f}".
                            format(get_cur_time(), self.ft_iter,
                                   self.ft_optimizer.param_groups[0]["lr"],
                                   sr_loss.cpu().data,
                                   bp_loss.cpu().data,
                                   align_lr_loss.cpu().data,
                                   total_loss.cpu().data))
                else:
                    print("===> {}: Iteration: {}, lr: {:.5f}, SR loss: {:.10f}, "
                          "Align LR loss: {:.10f}, Total loss: {:.10f}".
                          format(get_cur_time(), self.ft_iter,
                                 self.ft_optimizer.param_groups[0]["lr"],
                                 sr_loss.cpu().data,
                                 align_lr_loss.cpu().data,
                                 total_loss.cpu().data))


            if self.ft_iter == self.conf.ft_iter_step:
                self.lr_ft_stage *= 0.1
                print("Learning rate of FT stage is updated, now it's {:.10f}".format(
                    self.lr_ft_stage))
            if self.ft_iter == (self.conf.max_iter_ft - 1):
                print("Get maximum iteration number, start testing!")
                PSNR_value, sr_res = self.test(mode="ft")
                self.ft_test_PSNRs.append(PSNR_value)
                self.ft_test_iters.append(self.ft_iter // self.conf.test_step)
                return sr_res

    #### ---- Final test

    def final_test(self):
        """
        The difference between final test and normal test:
        A. geometry augmentation ensemble
        B. back-projection
        """

        outputs = []
        self.align_aggre_net.eval()
        # get the lr_lf and hr_lf
        lr_lf = self.dataloader_aggre.lf_lr
        hr_lf = self.dataloader_aggre.lf_hr
        U = lr_lf.shape[0]
        # get the central light fields
        view_start = (U - self.view_num) // 2
        lr_lf = lr_lf[view_start: view_start + self.view_num, view_start: view_start + self.view_num]
        hr_lf = hr_lf[view_start: view_start + self.view_num, view_start: view_start + self.view_num]
        hr_ref = hr_lf[self.refPos[0], self.refPos[1]]
        lr_ref = lr_lf[self.refPos[0], self.refPos[1]]

        ##### ----- self-ensemble
        # 4 <= k < 8, only vertical flip
        # 8 <= k < 12, only horizontal flip
        # 12 <= k < 16, vertical + horizontal
        for k in range(0, 16):
            if k < 4:
                test_input_lf = np.rot90(lr_lf, k, (2, 3))
                test_input_lf = np.rot90(test_input_lf, k, (0, 1))
            elif k < 8:
                # only vertical
                test_input_lf = np.flip(np.flip(lr_lf, 0), 2)
                test_input_lf = np.rot90(test_input_lf, k - 4, (2, 3))
                test_input_lf = np.rot90(test_input_lf, k - 4, (0, 1))
            elif k < 12:
                # only horizontal
                test_input_lf = np.flip(np.flip(lr_lf, 1), 3)
                test_input_lf = np.rot90(test_input_lf, k - 8, (2, 3))
                test_input_lf = np.rot90(test_input_lf, k - 8, (0, 1))
            else:
                # vertical and horizontal
                test_input_lf = np.flip(np.flip(lr_lf, 0), 2)
                test_input_lf = np.flip(np.flip(test_input_lf, 1), 3)
                test_input_lf = np.rot90(test_input_lf, k - 12, (2, 3))
                test_input_lf = np.rot90(test_input_lf, k - 12, (0, 1))

            # start test
            test_input_lf = test_input_lf.astype(float) / 255.0
            test_input_lf = torch.Tensor(test_input_lf.copy()).float().view(1, -1,
                                                                            test_input_lf.shape[2],
                                                                            test_input_lf.shape[3])
            aug_scale = self.scale
            # feed into cuda device, if needed
            if self.conf.use_cuda:
                test_input_lf = test_input_lf.cuda()
            else:
                test_input_lf = test_input_lf.cpu()
            # get the results
            trained_dict = self.align_aggre_net.module.state_dict()
            test_dict = self.align_aggre_net_test.state_dict()
            for_test = {k: v for k, v in trained_dict.items() if k in test_dict.keys()}
            test_dict.update(for_test)
            self.align_aggre_net_test.load_state_dict(test_dict)
            with torch.no_grad():
                # sr_cv = self.test_network(lr_lf, lr_disp)
                try:
                    _, _, sr_ref, _, _ = self.align_aggre_net_test(test_input_lf, aug_scale, self.conf.set_name)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                            _, _, sr_ref, _, _ = self.align_aggre_net_test(test_input_lf, aug_scale, self.conf.set_name)
                    else:
                        raise exception
            # sr_cv: 1, 1, H, W
            sr_ref = sr_ref.cpu().data
            sr_ref = sr_ref.squeeze()
            sr_ref = sr_ref.numpy().astype(np.float32)
            # transfer sr_cv back
            if k < 4:
                tmp_output = np.rot90(sr_ref, -k, (0, 1))
            elif k < 8:
                tmp_output = np.flip(np.rot90(sr_ref, 4 - k, (0, 1)), 0)
            elif k < 12:
                tmp_output = np.flip(np.rot90(sr_ref, 8 - k, (0, 1)), 1)
            else:
                tmp_output = np.flip(np.flip(np.rot90(sr_ref, 12 - k, (0, 1)), 1), 0)  # H, W

            # BP refinement
            t_tmp_output = torch.Tensor(tmp_output.copy()).unsqueeze(0).unsqueeze(0)
            t_lr_cv = torch.Tensor(lr_ref.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            if self.conf.use_cuda:
                t_tmp_output = t_tmp_output.cuda()
                t_lr_cv = t_lr_cv.cuda()
            for _ in range(self.conf.max_bp_iter):
                t_tmp_output = back_projection_refinement(t_lr_cv, t_tmp_output, self.scale)
            # back to numpy and [0, 1]
            t_tmp_output = t_tmp_output.squeeze().cpu().data.numpy().astype(np.float32)
            outputs.append(t_tmp_output)
        almost_final_sr = np.median(outputs, 0)
        almost_final_sr = transfer_img_to_uint8(almost_final_sr)

        ##### ----- Final back-projection refinement
        t_almost_final = torch.Tensor(almost_final_sr.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        t_lr_cv = torch.Tensor(lr_ref.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        if self.conf.use_cuda:
            t_almost_final = t_almost_final.cuda()
            t_lr_cv = t_lr_cv.cuda()
        for _ in range(self.conf.max_bp_iter):
            t_almost_final = back_projection_refinement(t_lr_cv, t_almost_final, self.scale)

        # back to numpy and [0, 255]
        t_almost_final = t_almost_final.squeeze().cpu().data.numpy().astype(np.float32)
        final_res = transfer_img_to_uint8(t_almost_final)

        # calculate PSNR
        psnr_ensemble = PSNR(almost_final_sr, hr_ref)
        psnr_final = PSNR(final_res, hr_ref)

        return psnr_ensemble, psnr_final, almost_final_sr, final_res

    def save_ckp(self, save_mode="final"):

        state = {"model_dict": self.align_aggre_net.module.state_dict()}
        if save_mode == "final":
            save_path = "{}/{}_model_final.pth".format(self.save_prefix,
                                                       self.save_name)
        else:
            save_path = "{}/{}_model_before_finetune.pth".format(self.save_prefix,
                                                                 self.save_name)
        torch.save(state, save_path)
        print("Checkpoint before finetuning saved to {}".format(save_path))

    def run(self):
        print("Start training AlignNet --------")
        self.trainer_align_stage()
        print("Freeze AlignNet, now train AggreNet ---------")
        aggre_trainer_sr_res = self.trainer_aggre_stage()
        if self.conf.record:
            print("Save the checkpoint before finetuning --------")
            self.save_ckp("before")
        print("Final testing for Aggre stage ----------")
        psnr_ensemble_aggre, psnr_final_aggre, ensemble_res_aggre, final_res_aggre = self.final_test()
        print("Start finetuning --------")
        ft_sr_res = self.ft_trainer()
        if self.conf.record:
            print("Save the checkpoint after finetuning ---------")
            self.save_ckp("final")
        print("Final testing ----------")
        psnr_ensemble_ft, psnr_final_ft, ensemble_res_ft, final_res_ft = self.final_test()

        results_dict = {"aggre_sr_res": aggre_trainer_sr_res,
                        "aggre_ensemble": ensemble_res_aggre,
                        "aggre_final_res": final_res_aggre,
                        "psnr_aggre_sr": self.aggre_test_PSNR[-1],
                        "psnr_aggre_ensemble": psnr_ensemble_aggre,
                        "psnr_aggre_final": psnr_final_aggre,
                        "ft_sr_res": ft_sr_res,
                        "ft_ensemble": ensemble_res_ft,
                        "ft_final": final_res_ft,
                        "psnr_ft_sr": self.ft_test_PSNRs[-1],
                        "psnr_ft_ensemble": psnr_ensemble_ft,
                        "psnr_ft_final": psnr_final_ft,
                        "vdsr_ref": self.vdsr_ref,
                        "psnr_vdsr": self.vdsr_PSNR}

        return results_dict


