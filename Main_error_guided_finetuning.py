"""
Main function for light field super-resolution with error-guided finetuning using our zero-shot scheme.

You can tune the running parameters with parser and the training parameters at Config_for_LFZSSR.
"""
import argparse
from tools.utils import *
from trainers.Class_error_guided_finetuning import *

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0", type=str,
                    help="gpu id (default: 0)")
parser.add_argument("--record", action="store_true",
                    help="Record?")
parser.add_argument("--scale", default=2, type=int,
                    help="scaling factor?")
parser.add_argument("--dataset", default="EPFL", type=str,
                    help="EPFL or HCI2")
parser.add_argument("--start", default=2, type=int,
                    help="Start index in dataset?")
parser.add_argument("--end", default=3, type=int,
                    help="End index in dataset?")
parser.add_argument("--gpu-num", default=1, type=int,
                    help="GPU number, for batch size setting")
parser.add_argument("--align-lr", default=1e-5, type=float,
                    help="Initial learning rate for warp stage, default: 1e-4")
parser.add_argument("--aggre-lr", default=1e-5, type=float,
                    help="Initial learning rate for fusion stage, default: 1e-4")
parser.add_argument("--ft-lr", default=1e-5, type=float,
                    help="Initial learning rate for finetune stage, default: 1e-4")
parser.add_argument("--disp-mode", default="ada", type=str,
                    help="Disparity range mode: adaptive (ada) or fixed (fix)")
parser.add_argument("--source", default="HFUT", type=str,
                    help="Source dataset: HFUT or SAE")

opt = parser.parse_args()

batch_size = opt.gpu_num
print(opt)

lf_set_name = opt.dataset
list_file_path = "./data/{}_list.txt".format(
    lf_set_name)

model_path = ""
fd = open(list_file_path, 'r')
name_list = [line.strip('\n') for line in fd.readlines()]
print("The name list of the tested light field dataset: '{}'".format(name_list))
PSNRs = np.zeros([7, len(name_list)])
count = 0

##################### hyperparameters start
save_name_dir = "ErrorGuided_Source_{}_Set_{}_Scale_{}_Bsz_{}_WarpLr{}_FuseLr{}_FtLr{}".format(opt.source,
                                                                                               opt.dataset,
                                                                                               opt.scale,
                                                                                               opt.gpu_num,
                                                                                               opt.align_lr,
                                                                                               opt.aggre_lr,
                                                                                               opt.ft_lr)
save_prefix = "./results/{}".format(save_name_dir)
configs = Config()
# Fixed parameters
configs.gpu_id = opt.gpus
configs.align_patch_size = 32
configs.batch_size = batch_size
configs.align_batch_size = batch_size
configs.ft_batch_size = batch_size
configs.weight_decay_aggre = 0.0
configs.zssr_bp_ratio = 0.5
configs.align_loss_weight = 0.1

configs.lr_align_stage = opt.align_lr
configs.lr_aggre_stage = opt.aggre_lr
configs.lr_ft_stage = opt.ft_lr
configs.train_time_test = opt.train_test

configs.view_num = 9
cv_uv = configs.view_num // 2
configs.refPos = [cv_uv, cv_uv]
configs.record = opt.record
configs.level_num = 50

# Adjustable parameters
if opt.scale == 2:
    configs.scale = opt.scale
    configs.patch_size = 64
    configs.max_iter_aggre = 4500
    configs.align_aggre_iter_step = 3000
elif opt.scale == 3:
    configs.scale = opt.scale
    configs.patch_size = 72
    configs.max_iter_aggre = 3500
    configs.align_aggre_iter_step = 2500
elif opt.scale == 4:
    configs.scale = opt.scale
    configs.patch_size = 72
    configs.max_iter_aggre = 3500
    configs.align_aggre_iter_step = 2500
else:
    raise Exception("Wrong scaling factor!")

ext_pre_model = "./pretrained/{}_pretrained_scale{}.pth".format(opt.source, configs.scale)

print(configs)

#################### hyperparameters end

for ind in range(opt.start, opt.end):
    lf_name = name_list[ind]
    print("The proessed light field: {}".format(lf_name))
    mat_path = "./data/{}_Y/{}.mat".format(
        lf_set_name, lf_name)
    save_name = lf_name

    hr_lf = loadmat(mat_path)["lf_hr"]
    hr_cv = hr_lf[4, 4]
    lr_cv = single_image_downscale(hr_cv, opt.scale)
    bic_lr_cv = single_image_upscale(lr_cv, opt.scale)
    psnr_bicubic = PSNR(bic_lr_cv, hr_cv)
    print("Bicubic interpolation after crop: {}".format(psnr_bicubic))

    if lf_set_name == "HCI1":
        configs.set_name = 'high'

    if opt.disp_mode == "fix":
        if configs.scale == 2:
            configs.disp_max = 2.0
        else:
            configs.disp_max = 1.5
    else:
        if lf_set_name == "HCI1":

            if lf_name == "rx_elephant" or lf_name == "rx_watch":
                configs.disp_max = 3.0 / configs.scale
            else:
                configs.disp_max = 2.5 / configs.scale
        elif lf_set_name == "HCI2":

            if lf_name == "HCI2_antinous" or lf_name == "HCI2_dishes" or lf_name == "HCI2_greek":
                configs.disp_max = 3.5 / configs.scale
            else:
                configs.disp_max = 2.5 / configs.scale
        else:
            configs.disp_max = 2.5 / configs.scale
    print("The disp mode is: {} and disp-max is {}".format(opt.disp_mode, configs.disp_max))

    ZSSRer = Error_guided_finetuning_SingleTargetView(lf_name=lf_name,
                                                      mat_path=mat_path,
                                                      conf=configs,
                                                      ext_model=ext_pre_model,
                                                      save_name=save_name,
                                                      save_prefix=save_prefix,
                                                      gpu_num=opt.gpu_num)
    results_dict = ZSSRer.run_three_stage()
    if opt.record:
        sio.savemat("{}/result_{}.mat".format(save_prefix, lf_name), results_dict)

    PSNRs[0, count] = results_dict["psnr_aggre_sr"]
    PSNRs[1, count] = results_dict["psnr_aggre_ensemble"]
    PSNRs[2, count] = results_dict["psnr_aggre_final"]
    PSNRs[3, count] = results_dict["psnr_ft_sr"]
    PSNRs[4, count] = results_dict["psnr_ft_ensemble"]
    PSNRs[5, count] = results_dict["psnr_ft_final"]
    PSNRs[6, count] = results_dict["psnr_vdsr"]
    count += 1
if opt.record:
    sio.savemat("{}/{}.mat".format(save_prefix, save_name_dir),
                {"PSNRs": PSNRs})

