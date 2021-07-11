"""
This script contains classes which define the dataloader for LFZSSR.
We generate LR light field patches for AlignNet and LLR-LR pairs for AggreNet as well as the finetuning stage.
Note that, although we implement the version with batch size larger than 1, it's better to set it as 1 considering
the I/O for speed.
"""

import torch
import random
import numpy as np
from tools.utils import *
from scipy.io import loadmat

class DataloaderForLFZSSRWithBatch:
    """
    The data should be with .mat format. In the file, there should be an HR light field.
    """
    def __init__(self,
                 mat_path,
                 refPos,
                 scale,
                 view_num,
                 min_scale=0.5,
                 max_scale=1,
                 patch_size=64,
                 batch_size=1,
                 scale_aug=True,
                 random_flip_vertical=True,
                 random_flip_horizontal=True,
                 random_rotation=True):
        # read the data first
        self.mat_data = loadmat(mat_path)
        self.lf_hr = self.mat_data['lf_hr']
        self.lf_lr = LF_downscale(self.lf_hr, scale)
        # scale factor for super-resolution
        self.scale = scale
        self.refPos = refPos
        # scales for augmentation
        self.scale_aug = scale_aug
        self.min_scale = min_scale
        self.max_scale = max_scale
        # the processed view number
        self.view_num = view_num
        # patch size
        self.patch_size = patch_size
        self.batch_size = batch_size
        # random augmentation
        self.random_flip_vertical = random_flip_vertical
        self.random_flip_horizontal = random_flip_horizontal
        self.random_rotation = random_rotation
        # get some parameters
        self.ori_view_num = self.lf_hr.shape[0]
        self.lr_height = self.lf_lr.shape[2]
        self.lr_width = self.lf_lr.shape[3]

        # refine the min scale
        self.min_scale = min(max(self.min_scale, self.patch_size / min(self.lr_height, self.lr_width)), self.max_scale)
        # bicubic interpolation
        self.resize = bicubic_imresize()

    def get_patch(self):
        # get the wanted views
        view_start = (self.ori_view_num - self.view_num) // 2
        lf_lr = self.lf_lr[view_start: view_start + self.view_num, view_start: view_start + self.view_num]

        # get the HR father
        hr_father = torch.Tensor(lf_lr.astype(np.float32) / 255.0).contiguous().view(1, -1, self.lr_height, self.lr_width)
        # init aug scale
        aug_scale = 1.0

        if self.scale_aug:
            aug_scale = random.uniform(self.min_scale, self.max_scale)
            hr_father = self.resize(hr_father, aug_scale) # HR father
        father_hei, father_wid = hr_father.shape[2], hr_father.shape[3]
        father_hei = father_hei - (father_hei % self.scale)
        father_wid = father_wid - (father_wid % self.scale)
        # modcrop
        hr_father = hr_father[:, :, :father_hei, :father_wid]
        lr_son = self.resize(hr_father, 1.0/self.scale) # LR son
        son_hei, son_wid = lr_son.shape[2], lr_son.shape[3]

        # get them back to numpy
        hr_father = hr_father.view(self.view_num, -1, father_hei, father_wid)
        hr_father = hr_father.numpy()
        lr_son = lr_son.view(self.view_num, -1, son_hei, son_wid)
        lr_son = lr_son.numpy()

        # avoid the problem that size of lr_son is smaller than patch size
        if father_hei <= self.patch_size and father_wid <= self.patch_size:
            hr_patch = np.zeros([self.batch_size, self.view_num, self.view_num,
                                 father_hei, father_wid], dtype=np.float32)
            lr_patch = np.zeros([self.batch_size, self.view_num, self.view_num,
                                 son_hei, son_wid], dtype=np.float32)
        elif father_hei > self.patch_size and father_wid <= self.patch_size:
            hr_patch = np.zeros([self.batch_size, self.view_num, self.view_num,
                                 self.patch_size, father_wid], dtype=np.float32)
            lr_patch = np.zeros([self.batch_size, self.view_num, self.view_num,
                                 self.patch_size // self.scale,
                                 son_wid], dtype=np.float32)
        elif father_hei <= self.patch_size and father_wid > self.patch_size:
            hr_patch = np.zeros([self.batch_size, self.view_num, self.view_num,
                                 father_hei, self.patch_size], dtype=np.float32)
            lr_patch = np.zeros([self.batch_size, self.view_num, self.view_num,
                                 son_hei, self.patch_size // self.scale], dtype=np.float32)
        else:
            hr_patch = np.zeros([self.batch_size, self.view_num, self.view_num,
                                 self.patch_size, self.patch_size], dtype=np.float32)
            lr_patch = np.zeros([self.batch_size, self.view_num, self.view_num,
                                 self.patch_size // self.scale,
                                 self.patch_size // self.scale], dtype=np.float32)

        for i in range(self.batch_size):
            if father_hei <= self.patch_size and father_wid <= self.patch_size:
                hr_patch[i] = hr_father
                lr_patch[i] = lr_son
            elif father_hei > self.patch_size and father_wid <= self.patch_size:
                x = random.randrange(0, father_hei - self.patch_size, self.scale)
                hr_patch[i] = hr_father[:, :, x:x+self.patch_size, :]
                lr_patch[i] = lr_son[:, :, x // self.scale:(x+self.patch_size) // self.scale, :]
            elif father_hei <= self.patch_size and father_wid > self.patch_size:
                y = random.randrange(0, father_wid - self.patch_size, self.scale)
                hr_patch[i] = hr_father[:, :, :, y:y+self.patch_size]
                lr_patch[i] = lr_son[:, :, :, y // self.scale: (y+self.patch_size) // self.scale]
            else:
                x = random.randrange(0, father_hei - self.patch_size, self.scale)
                y = random.randrange(0, father_wid - self.patch_size, self.scale)
                hr_patch[i] = hr_father[:, :, x: x + self.patch_size, y: y + self.patch_size]
                lr_patch[i] = lr_son[:, :, x // self.scale: (x + self.patch_size) // self.scale,
                              y // self.scale: (y + self.patch_size) // self.scale]

        # 4D augmentation
        if self.random_flip_vertical and np.random.rand(1) > 0.5:
            hr_patch = np.flip(np.flip(hr_patch, 1), 3)
            lr_patch = np.flip(np.flip(lr_patch, 1), 3)
        if self.random_flip_horizontal and np.random.rand(1) > 0.5:
            hr_patch = np.flip(np.flip(hr_patch, 2), 4)
            lr_patch = np.flip(np.flip(lr_patch, 2), 4)
        if self.random_rotation:
            r_ang = np.random.randint(1, 5)
            hr_patch = np.rot90(hr_patch, r_ang, (3, 4))
            hr_patch = np.rot90(hr_patch, r_ang, (1, 2))
            lr_patch = np.rot90(lr_patch, r_ang, (3, 4))
            lr_patch = np.rot90(lr_patch, r_ang, (1, 2))
        # return torch.Tensor
        hr_ref_patch = hr_patch[:, self.refPos[0], self.refPos[1], :, :]
        hr_ref_patch = torch.Tensor(hr_ref_patch.copy()).contiguous()
        hr_ref_patch = hr_ref_patch.unsqueeze(1)

        lr_ref_patch = lr_patch[:, self.refPos[0], self.refPos[1], :, :]
        lr_ref_patch = torch.Tensor(lr_ref_patch.copy()).contiguous()
        lr_ref_patch = lr_ref_patch.unsqueeze(1)

        hr_patch = torch.Tensor(hr_patch.copy()).contiguous()
        hr_patch = hr_patch.view(self.batch_size, -1, hr_patch.shape[3], hr_patch.shape[4])
        lr_patch = torch.Tensor(lr_patch.copy()).contiguous()
        lr_patch = lr_patch.view(self.batch_size, -1, lr_patch.shape[3], lr_patch.shape[4])
        return lr_patch, hr_patch, lr_ref_patch, hr_ref_patch, aug_scale


class DataloaderForAlignNetWithBatch:
    def __init__(self,
                 mat_path,
                 refPos,
                 scale,
                 view_num,
                 patch_size=64,
                 batch_size=1,
                 random_flip_vertical=True,
                 random_flip_horizontal=True,
                 random_rotation=True):
        # read the data first
        self.mat_data = loadmat(mat_path)
        self.lf_hr = self.mat_data['lf_hr']
        self.lf_lr = LF_downscale(self.lf_hr, scale)
        # scale factor for super-resolution
        self.refPos = refPos
        # the processed view number
        self.view_num = view_num
        # patch size
        self.patch_size = patch_size
        self.batch_size = batch_size
        # random augmentation
        self.random_flip_vertical = random_flip_vertical
        self.random_flip_horizontal = random_flip_horizontal
        self.random_rotation = random_rotation
        # get some parameters
        self.ori_view_num = self.lf_hr.shape[0]
        self.lr_height = self.lf_lr.shape[2]
        self.lr_width = self.lf_lr.shape[3]

    def get_patch(self):
        # get the wanted views
        view_start = (self.ori_view_num - self.view_num) // 2
        lf_lr = self.lf_lr[view_start: view_start + self.view_num, view_start: view_start + self.view_num]
        lf_lr = lf_lr.astype(np.float32) / 255.0

        lf_patch = np.zeros([self.batch_size, self.view_num, self.view_num,
                             self.patch_size, self.patch_size], dtype=np.float32)
        for i in range(self.batch_size):
            x = random.randrange(0, self.lr_height - self.patch_size)
            y = random.randrange(0, self.lr_width - self.patch_size)
            lf_patch[i] = lf_lr[:, :, x: x + self.patch_size, y: y + self.patch_size]

        # 4D augmentation
        if self.random_flip_vertical and np.random.rand(1) > 0.5:
            lf_patch = np.flip(np.flip(lf_patch, 1), 3)
        if self.random_flip_horizontal and np.random.rand(1) > 0.5:
            lf_patch = np.flip(np.flip(lf_patch, 2), 4)
        if self.random_rotation:
            r_ang = np.random.randint(1, 5)
            lf_patch = np.rot90(lf_patch, r_ang, (3, 4))
            lf_patch = np.rot90(lf_patch, r_ang, (1, 2))
        # return torch.Tensor

        lf_patch = torch.Tensor(lf_patch.copy()).contiguous()
        lf_patch = lf_patch.view(self.batch_size, -1, self.patch_size, self.patch_size)

        return lf_patch
