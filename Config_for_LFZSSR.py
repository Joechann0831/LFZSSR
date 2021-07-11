"""
Parameters for LFZSSR.
"""

class Config:

    scale = 2
    view_num = 9
    refPos = [4, 4]
    patch_size = 64
    align_patch_size = 64
    weight_decay = 0.0
    weight_decay_aggre = 0.0
    aggre_batch_size = 1
    align_batch_size = 1
    ft_batch_size = 1
    random_seed = None

    ######## devices
    use_cuda = True
    gpu_id = '0'

    ######## early stop and test
    max_iters = 20000
    test_step = 50
    min_check = 5
    min_learning_rate = 1e-6

    ######## record
    record = False
    display_loss_step = 20

    ######## For back-projection refinement
    max_bp_iter = 10
    scale_aug = True
    pad_size = 12

    ############### -----------------------

    # For VDSR
    vdsr_model_path = './pretrained/VDSR_model.pth'

    # For AlignNet
    disp_max = 2.0
    level_num = 64

    # For finetune
    align_loss_weight = 0.1
    set_name = "low" # "high" or "low"
    zssr_bp_ratio = 0.5

    # for scheduler learning
    lr_align_stage = 1e-4
    lr_aggre_stage = 1e-4
    lr_ft_stage = 1e-4

    max_iter_aggre = 3500
    max_iter_ft = 3500
    align_aggre_iter_step = 3000
    ft_iter_step = 2500



