"""
    utils.py is used to define useful functions
"""

import math
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import os
import h5py
import argparse

def get_time_gpu():
    torch.cuda.synchronize()
    return time.time()


def get_time_gpu_str():
    torch.cuda.synchronize()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class StoreAsArray(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


class bicubic_imresize(nn.Module):
    """
    An pytorch implementation of imresize function in MATLAB with bicubic kernel.
    """
    def __init__(self):
        super(bicubic_imresize, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale, cuda_flag):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)
        if cuda_flag:
            x0 = x0.cuda()
            x1 = x1.cuda()

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        if cuda_flag:
            indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
            indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
        else:
            indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
            indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        if cuda_flag:
            indice0 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice0),
                                torch.FloatTensor([in_size[0]]).cuda()).unsqueeze(0)
            indice1 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice1),
                                torch.FloatTensor([in_size[1]]).cuda()).unsqueeze(0)
        else:
            indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0),
                                torch.FloatTensor([in_size[0]])).unsqueeze(0)
            indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1),
                                torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1 / 4):
        [b, c, h, w] = input.shape
        output_size = [b, c, int(h * scale), int(w * scale)]
        cuda_flag = input.is_cuda

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale, cuda_flag)
        weight0 = weight0.squeeze(0)
        indice0 = indice0.squeeze(0).long()
        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        weight1 = weight1.squeeze(0)

        indice1 = indice1.squeeze(0).long()
        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = torch.sum(out, dim=3).permute(0, 1, 3, 2)
        return out
#
def LF_downscale(input_lf, scale):
    """
    Downscale the input light field.
    :param input_lf: [U,V,X,Y], dtype: uint8
    :param scale: should be larger than 1
    :return: resized light field
    """
    U, H, W = input_lf.shape[1], input_lf.shape[2], input_lf.shape[3]
    H = H - H % scale
    W = W - W % scale
    h = H // scale
    w = W // scale
    resize = bicubic_imresize()
    input_lf = input_lf[:, :, :H, :W]
    input_lf = torch.Tensor(input_lf.astype(np.float32) / 255.0).contiguous().view(1, -1, H, W)
    resize_lf = resize(input_lf, 1.0/scale)  # [1, U*V, H/s, W/s]
    resize_lf = resize_lf.view(-1, U, h, w)
    resize_lf = torch.round(resize_lf * 255.0)
    resize_lf = torch.clamp(resize_lf, 0.0, 255.0)
    resize_lf = resize_lf.numpy().astype(np.uint8)
    return resize_lf

def LF_downscale_RGB(input_lf, scale):
    """
    LF_downscale with RGB light field.
    :param input_lf: [C,U,V,X,Y], dtype: uint8
    :param scale: should be larger than 1
    :return: resized RGB light field
    """
    C, U, H, W = input_lf.shape[0], input_lf.shape[1], input_lf.shape[3], input_lf.shape[4]
    H = H - H % scale
    W = W - W % scale
    h = H // scale
    w = W // scale
    resize = bicubic_imresize()
    input_lf = input_lf[:, :, :, :H, :W]
    input_lf = torch.Tensor(input_lf.astype(np.float32) / 255.0).contiguous().view(1, -1, H, W)
    resize_lf = resize(input_lf, 1.0/scale)  # [1, C*U*V, H/s, W/s]
    resize_lf = resize_lf.view(C, U, -1, h, w)
    resize_lf = torch.round(resize_lf * 255.0)
    resize_lf = torch.clamp(resize_lf, 0.0, 255.0)
    resize_lf = resize_lf.numpy().astype(np.uint8)
    return resize_lf

def back_projection_refinement(lr_img, sr_res, scale):
    """
    Back-projection refinement for further improvement of SR results and avoid possible in-consistency
    between LR and HR results.
    :param lr_img: [1, 1, h, w], FloatTensor
    :param sr_res: [1, 1, H, W], FloatTensor
    :param scale: the scaling factor \alpha
    :return: refined result
    """
    resizer = bicubic_imresize()
    refined_res = sr_res + resizer(lr_img - resizer(sr_res, 1.0/scale), scale)
    return refined_res

def PSNR(pred, gt, shave_border=0):
    """
    PSNR function.
    :param pred:            uint8, predicted result
    :param gt:              uint8, ground truth
    :param shave_border:    int, crop the border to calculate PSNR
    :return: PSNR value in dB.
    """
    pred = pred.astype(float)
    gt = gt.astype(float)
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def transfer_img_to_uint8(img):
    """
    Transfer float tensor within [0,1] to uint8
    :param img: Float, [0,1]
    :return: uint8 version
    """
    img = img * 255.0
    img = np.clip(img, 0.0, 255.0)
    img = np.uint8(np.around(img))
    return img

def colorize(y, ycbcr):
    """
    Colorize a grayscale image with super-resolved y and interpolated Cb and Cr.
    :param y:              Super-resolved LR Y.
    :param ycbcr:          Interpolated YCbCr using Bicubic interpolation.
    :return: img:          Colorized result.
    """
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

def modcrop(imgs, scale):
    """
    Mod and crop. The function is always used before resizing.
    :param imgs: the channel should be 1 or 3.
    :param scale: should be an integer.
    :return: cropped_img
    """
    if len(imgs.shape) == 2:
        img_row = imgs.shape[0]
        img_col = imgs.shape[1]
        cropped_row = img_row - img_row % scale
        cropped_col = img_col - img_col % scale
        cropped_img = imgs[:cropped_row, :cropped_col]
    elif len(imgs.shape) == 3:
        img_row = imgs.shape[0]
        img_col = imgs.shape[1]
        cropped_row = img_row - img_row % scale
        cropped_col = img_col - img_col % scale
        cropped_img = imgs[:cropped_row, :cropped_col, :]
    else:
        raise IOError('Img Channel > 3.')

    return cropped_img

def lf_modcrop(lf, scale):
    """
    Modcrop the input light field.
    :param lf: [U,V,X,Y]
    :param scale: int
    :return: cropped light field.
    """
    [U, V, X, Y] = lf.shape
    x = X - (X % scale)
    y = Y - (Y % scale)
    output = np.zeros([U, V, x, y])
    for u in range(0, U):
        for v in range(0, V):
            sub_img = lf[u,v]
            output[u,v] = modcrop(sub_img, scale)
    return output

def single_image_downscale(hr_img, scale):
    """
    Downscale a single uint8 image.
    :param hr_img: [H, W], dtype: uint8
    :param scale: should be larger than 1.
    :return: downsampeld hr_img
    """
    hr_img = modcrop(hr_img, scale)
    resize = bicubic_imresize()
    hr_img = torch.Tensor(hr_img.astype(np.float32) / 255.0).contiguous().unsqueeze(0).unsqueeze(0)
    lr_img = resize(hr_img, 1.0/scale)
    lr_img = torch.clamp(torch.round(lr_img * 255.0), 0.0, 255.0)
    lr_img = lr_img.squeeze().numpy().astype(np.uint8) # [h, w]
    return lr_img

def single_image_upscale(lr_img, scale):
    """
    Upscale a single uint8 image.
    :param lr_img: [h, w], dtype: uint8
    :param scale: should be larger than 1.
    :return: upsampeld lr_img
    """
    resize = bicubic_imresize()
    lr_img = torch.Tensor(lr_img.astype(np.float32) / 255.0).contiguous().unsqueeze(0).unsqueeze(0)
    hr_img = resize(lr_img, scale)
    hr_img = torch.clamp(torch.round(hr_img * 255.0), 0.0, 255.0)
    hr_img = hr_img.squeeze().numpy().astype(np.uint8) # [h, w]
    return hr_img

def img_rgb2ycbcr(img):
    # the input image data format should be uint8
    if not len(img.shape) == 3:
        raise IOError('Img channle is not 3')
    if not img.dtype == 'uint8':
        raise IOError('Img should be uint8')
    img = img/255.0
    img_ycbcr = np.zeros(img.shape, 'double')
    img_ycbcr[:, :, 0] = 65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2] + 16
    img_ycbcr[:, :, 1] = -37.797 * img[:, :, 0] - 74.203 * img[:, :, 1] + 112 * img[:, :, 2] + 128
    img_ycbcr[:, :, 2] = 112 * img[:, :, 0] - 93.786 * img[:, :, 1] - 18.214 * img[:, :, 2] + 128
    img_ycbcr = np.round(img_ycbcr)
    img_ycbcr = np.clip(img_ycbcr,0,255)
    img_ycbcr = np.uint8(img_ycbcr)
    return img_ycbcr

def img_ycbcr2rgb(im):
    # the input image data format should be uint8
    if not len(im.shape) == 3:
        raise IOError('Img channle is not 3')
    if not im.dtype == 'uint8':
        raise IOError('Img should be uint8')
    im_YCrCb = np.zeros(im.shape, 'double')
    im_YCrCb = im * 1.0
    tmp = np.zeros(im.shape, 'double')
    tmp[:, :, 0] = im_YCrCb[:, :, 0] - 16.0
    tmp[:, :, 1] = im_YCrCb[:, :, 1] - 128.0
    tmp[:, :, 2] = im_YCrCb[:, :, 2] - 128.0
    im_my = np.zeros(im.shape, 'double')
    im_my[:, :, 0] = 0.00456621 * tmp[:, :, 0] + 0.00625893 * tmp[:, :, 2]
    im_my[:, :, 1] = 0.00456621 * tmp[:, :, 0] - 0.00153632 * tmp[:, :, 1] - 0.00318811 * tmp[:, :, 2]
    im_my[:, :, 2] = 0.00456621 * tmp[:, :, 0] + 0.00791071 * tmp[:, :, 1]
    im_my = im_my * 255
    im_my = np.round(im_my)
    im_my = np.clip(im_my, 0, 255)
    im_my = np.uint8(im_my)
    return im_my

#############################################################################################################
# The following functions are used for warping.
# To facilitate the speed during training and the memory during inferencing, we have many variants
# here. warp_no_range() is the original one. We find that the function torch.arange is time consuming,
# so we have some variants using pre-defined arange variables. Moreover, we also have "parallel" or
# "serial" to distinguish the warping functions which process different views simutaneously or sequentially.
#############################################################################################################

def warp(x, flo, arange_spatial, padding_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    x and flo should be inside the same device (CPU or GPU)

    """
    B, C, H, W = x.size()
    xx = arange_spatial.view(1, -1).repeat(H, 1)
    yy = arange_spatial.view(-1, 1).repeat(1, W)

    xx = xx.view(1, 1, H, W)
    yy = yy.view(1, 1, H, W)
    grid = torch.cat((xx, yy), 1).float()  # [1, 2, H, W]
    grid = grid.repeat(B, 1, 1, 1)  # [B, 2, H, W]

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgridx = vgrid[:, 0, :, :].clone().unsqueeze(1)
    vgridy = vgrid[:, 1, :, :].clone().unsqueeze(1)
    vgridx = 2.0 * vgridx / max(W - 1, 1) - 1.0
    vgridy = 2.0 * vgridy / max(H - 1, 1) - 1.0

    vgrid = torch.cat([vgridx, vgridy], dim=1)
    vgrid = vgrid.permute(0, 2, 3, 1).contiguous()
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode=padding_mode)
    return output


def warp_to_ref_view_parallel(input_lf, disparity, refPos,
                              arange_angular, arange_spatial, padding_mode="zeros"):
    """
    This is the function used for warping a light field to the reference view.
    Unlike warp_to_central_view_lf, we do not use for circle here, we use parallel computation.
    :param input_lf: [B, U*V, H, W]
    :param disparity: [B, 2, H, W]
    :param refPos: u and v coordinates of the reference view point
    :param padding_mode: mode for padding
    :return: return the warped images
    """
    B, UV, H, W = input_lf.shape
    U = int(math.sqrt(float(UV)))
    ref_u = refPos[1] # horizontal angular coordinate
    ref_v = refPos[0] # vertical angular coordinate

    # for speed
    uu = arange_angular.view(1, -1).repeat(U, 1) # u direction, X
    vv = arange_angular.view(-1, 1).repeat(1, U) # v direction, Y

    uu = uu.view(1, -1, 1, 1, 1) - ref_u
    vv = vv.view(1, -1, 1, 1, 1) - ref_v
    deta_uv = torch.cat([uu, vv], dim=2)  # [1, U*V, 2, 1, 1]
    deta_uv = deta_uv.repeat(B, 1, 1, 1, 1)  # [B, U*V, 2, 1, 1]
    if input_lf.is_cuda:
        deta_uv = deta_uv.cuda()
    deta_uv = deta_uv.float()
    ## generate the full disparity maps
    full_disp = disparity.unsqueeze(1) # [B, 1, 2, H, W]
    full_disp = full_disp.repeat(1, UV, 1, 1, 1) # [B, U*V, 2, H, W]
    full_disp = full_disp * deta_uv # [B, U*V, 2, H, W]
    ## warp
    input_lf = input_lf.view(-1, 1, H, W) # [B*U*V, 1, H, W]
    full_disp = full_disp.view(-1, 2, H, W) # [B*U*V, 2, H, W]
    warped_lf = warp(input_lf, full_disp, arange_spatial, padding_mode=padding_mode) # [B*U*V, 1, H, W]
    warped_lf = warped_lf.view(B, -1, H, W) # [B, U*V, H, W]
    return warped_lf

def warp_no_range(x, flo, padding_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    x and flo should be inside the same device (CPU or GPU)

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

    xx = xx.view(1, 1, H, W)
    yy = yy.view(1, 1, H, W)
    grid = torch.cat((xx, yy), 1).float()  # [1, 2, H, W]
    grid = grid.repeat(B, 1, 1, 1)  # [B, 2, H, W]

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgridx = vgrid[:, 0, :, :].clone().unsqueeze(1)
    vgridy = vgrid[:, 1, :, :].clone().unsqueeze(1)
    vgridx = 2.0 * vgridx / max(W - 1, 1) - 1.0
    vgridy = 2.0 * vgridy / max(H - 1, 1) - 1.0

    vgrid = torch.cat([vgridx, vgridy], dim=1)

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode=padding_mode)

    return output

def warp_to_ref_view_parallel_no_range(input_lf, disparity, refPos, padding_mode="zeros"):
    """
    This is the function used for warping a light field to the reference view.
    Unlike warp_to_central_view_lf, we do not use for circle here, we use parallel computation.
    :param input_lf: [B, U*V, H, W]
    :param disparity: [B, 2, H, W]
    :param refPos: u and v coordinates of the reference view point
    :param padding_mode: mode for padding
    :return: return the warped images
    """
    B, UV, H, W = input_lf.shape
    U = int(math.sqrt(float(UV)))
    ref_u = refPos[1] # horizontal angular coordinate
    ref_v = refPos[0] # vertical angular coordinate
    ## generate angular grid
    uu = torch.arange(0, U).view(1, -1).repeat(U, 1) # u direction, X
    vv = torch.arange(0, U).view(-1, 1).repeat(1, U) # v direction, Y

    uu = uu.view(1, -1, 1, 1, 1) - ref_u
    vv = vv.view(1, -1, 1, 1, 1) - ref_v
    deta_uv = torch.cat([uu, vv], dim=2)  # [1, U*V, 2, 1, 1]
    deta_uv = deta_uv.repeat(B, 1, 1, 1, 1)  # [B, U*V, 2, 1, 1]
    if input_lf.is_cuda:
        deta_uv = deta_uv.cuda()
    deta_uv = deta_uv.float()
    ## generate the full disparity maps
    full_disp = disparity.unsqueeze(1) # [B, 1, 2, H, W]
    full_disp = full_disp.repeat(1, UV, 1, 1, 1) # [B, U*V, 2, H, W]
    full_disp = full_disp * deta_uv # [B, U*V, 2, H, W]
    ## warp
    input_lf = input_lf.view(-1, 1, H, W) # [B*U*V, 1, H, W]
    full_disp = full_disp.view(-1, 2, H, W) # [B*U*V, 2, H, W]
    warped_lf = warp_no_range(input_lf, full_disp, padding_mode=padding_mode) # [B*U*V, 1, H, W]
    warped_lf = warped_lf.view(B, -1, H, W) # [B, U*V, H, W]
    return warped_lf

def warp_double_range(x, flo, arange_spatial_x, arange_spatial_y, padding_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    x and flo should be inside the same device (CPU or GPU)

    """
    B, C, H, W = x.size()

    # for speed
    xx = arange_spatial_x.view(1, -1).repeat(H, 1) # [H, W]
    yy = arange_spatial_y.view(-1, 1).repeat(1, W) # [H, W]

    xx = xx.view(1, 1, H, W)
    yy = yy.view(1, 1, H, W)
    grid = torch.cat((xx, yy), 1).float() # [1, 2, H, W]
    grid = grid.repeat(B, 1, 1, 1) # [B, 2, H, W]


    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    vgridx = vgrid[:, 0, :, :].clone().unsqueeze(1)
    vgridy = vgrid[:, 1, :, :].clone().unsqueeze(1)
    vgridx = 2.0 * vgridx / max(W - 1, 1) - 1.0
    vgridy = 2.0 * vgridy / max(H - 1, 1) - 1.0

    vgrid = torch.cat([vgridx, vgridy], dim=1)
    vgrid = vgrid.permute(0, 2, 3, 1).contiguous()
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode=padding_mode)

    return output

def warp_to_ref_view_parallel_double_range(input_lf, disparity, refPos,
                                           arange_angular, arange_spatial_x,
                                           arange_spatial_y, padding_mode="zeros"):
    """
    This is the function used for warping a light field to the reference view.
    Unlike warp_to_central_view_lf, we do not use for circle here, we use parallel computation.
    :param input_lf: [B, U*V, H, W]
    :param disparity: [B, 2, H, W]
    :param refPos: u and v coordinates of the reference view point
    :param padding_mode: mode for padding
    :return: return the warped images
    """
    B, UV, H, W = input_lf.shape
    U = int(math.sqrt(float(UV)))
    ref_u = refPos[1] # horizontal angular coordinate
    ref_v = refPos[0] # vertical angular coordinate

    # for speed
    uu = arange_angular.view(1, -1).repeat(U, 1) # u direction, X
    vv = arange_angular.view(-1, 1).repeat(1, U) # v direction, Y

    uu = uu.view(1, -1, 1, 1, 1) - ref_u
    vv = vv.view(1, -1, 1, 1, 1) - ref_v
    deta_uv = torch.cat([uu, vv], dim=2) # [1, U*V, 2, 1, 1]
    deta_uv = deta_uv.repeat(B, 1, 1, 1, 1) # [B, U*V, 2, 1, 1]


    if input_lf.is_cuda:
        deta_uv = deta_uv.cuda()
    deta_uv = deta_uv.float()
    ## generate the full disparity maps
    full_disp = disparity.unsqueeze(1) # [B, 1, 2, H, W]
    full_disp = full_disp.repeat(1, UV, 1, 1, 1) # [B, U*V, 2, H, W]
    full_disp = full_disp * deta_uv # [B, U*V, 2, H, W]
    ## warp
    input_lf = input_lf.view(-1, 1, H, W) # [B*U*V, 1, H, W]
    full_disp = full_disp.view(-1, 2, H, W) # [B*U*V, 2, H, W]
    warped_lf = warp_double_range(input_lf, full_disp, arange_spatial_x,
                                  arange_spatial_y, padding_mode=padding_mode) # [B*U*V, 1, H, W]
    warped_lf = warped_lf.view(B, -1, H, W) # [B, U*V, H, W]
    return warped_lf

def warp_to_ref_view_serial_no_range(input_lf, disparity, refPos, padding_mode="zeros"):
    """
    This is the function used for warping a light field to the reference view.
    Serial and no range are used for inference to avoid possible OOM and changeable resolution.
    :param input_lf: [B, U*V, H, W]
    :param disparity: [B, 2, H, W]
    :param refPos: u and v coordinates of the reference view point
    :param padding_mode: mode for padding
    :return: return the warped images
    """
    B, UV, H, W = input_lf.shape
    U = int(math.sqrt(UV))
    V = U
    ref_u = refPos[0]
    ref_v = refPos[1]

    input_lf = input_lf.view(-1, U, V, H, W)
    warped_ref_view = []

    for u in range(U):
        for v in range(V):
            disparity_x = disparity[:, 0, :, :].clone().unsqueeze(1)
            disparity_y = disparity[:, 1, :, :].clone().unsqueeze(1)
            deta_u = v - ref_v
            deta_v = u - ref_u

            disparity_x = deta_u * disparity_x
            disparity_y = deta_v * disparity_y
            disparity_copy = torch.cat([disparity_x, disparity_y], dim=1)

            sub_img = input_lf[:, u, v, :, :].clone().unsqueeze(1)

            warped_img = warp_no_range(sub_img, disparity_copy, padding_mode=padding_mode)
            warped_ref_view.append(warped_img)
    warped_ref_view = torch.cat(warped_ref_view, dim=1)
    return warped_ref_view

def warp_to_central_view_lf(input_lf, disparity, padding_mode="zeros"):
    """
    Warp the input light field to the central view using disparity map.
    :param input_lf: [B, U*V, H, W]
    :param disparity: [B, 2, H, W], disparity map of central view
    :param padding_mode: mode for padding, "zeros", "reflection" or "border"
    :return: return the warped central view images
    """
    B, UV, H, W = input_lf.size()
    U = int(math.sqrt(float(UV)))
    V = U
    mid_u = U // 2
    mid_v = V // 2

    input_lf = input_lf.view(-1, U, V, H, W)
    warped_central_view = []

    for u in range(U):
        for v in range(V):
            # disparity_copy = disparity.clone()
            disparity_x = disparity[:, 0, :, :].clone().unsqueeze(1)
            disparity_y = disparity[:, 1, :, :].clone().unsqueeze(1)
            deta_u = v - mid_v
            deta_v = u - mid_u

            disparity_x = deta_u * disparity_x
            disparity_y = deta_v * disparity_y
            disparity_copy = torch.cat([disparity_x, disparity_y], dim=1)

            sub_img = input_lf[:, u, v, :, :].clone().unsqueeze(1)

            warped_img = warp_no_range(sub_img, disparity_copy, padding_mode=padding_mode)
            warped_central_view.append(warped_img)
    warped_central_view = torch.cat(warped_central_view, dim=1)
    return warped_central_view

