# Descriptions about hyper-parameters

This file gives descriptions and directions about the hyper-parameters defined in [configs/Config_for_LFZSSR.py](https://github.com/Joechann0831/LFZSSR/blob/master/configs/Config_for_LFZSSR.py) and [configs/Config_error_guided_finetuning.py](https://github.com/Joechann0831/LFZSSR/blob/master/configs/Config_error_guided_finetuning.py).

### Common hyper-parameters

- **scale**: the scaling factor.
- **view_num**: the number of views in the input light field image.
- **refPos**: with format [u,v], defines the angular position of the reference view.
- **patch_size**: patch size used for training AggreNet as well as finetuning, it may be different for different scaling factors which is pre-defined in main scripts [Main_LFZSSR.py](https://github.com/Joechann0831/LFZSSR/blob/master/Main_LFZSSR.py) or [Main_error_guided_finetuning.py](https://github.com/Joechann0831/LFZSSR/blob/master/Main_error_guided_finetuning.py).
- **align_patch_size**: patch size used for training AlignNet.
- **weight_decay/ weight_decay_aggre**: weight decay for different stages, default is zero.
- **aggre_batch_size/ align_batch_size/ ft_batch_size**: batch size for training, it's always the same with the number of used GPUs and better be set as 1 for fast training.
- **random_seed**: random seed, "None" for random generation.
- **max_iters**: maximum iteration number during training.
- **test_step**: step for testing.
- **min_learning_rate**: minimum learning rate, if the learning rate reaches this value, the training will stop.
- **record**: set to True if you want to record the log and results.
- **display_loss_step**: step for displaying and recording.
- **max_bp_iter**: maximum iteration number for back-projection refinement.
- **pad_size**: padding size during PSV generation and alignment.
- **disp_max**: maximum disparity value, it should be different with different input light fields. For example, we set different values for disp_max in [Main_error_guided_finetuning.py](https://github.com/Joechann0831/LFZSSR/blob/master/Main_error_guided_finetuning.py), you can tune it to fit your own data.
- **level_num**: the level number used for PSV generation, default is 50, if you want to make the alignment better, you can make it larger. However, larger level_num will lead to severe memory and computation cost increasing.
- **align_loss_weight**: loss weight for align loss, i.e., $\gamma_{2}$ in the paper.
- **set_name**: dataset name for testing. "low" for low resolution input and "high" for high-resolution input. For the latter, we will use serial PSV generator to avoid possible OOM.
- **zssr_bp_ratio**: loss weight for back-projection loss, i.e., $\gamma_{1}$ in the paper.
- **lr_align_stage/ lr_aggre_stage/ lr_ft_stage**: init learning rate for each stage.
- **max_iter_aggre/ max_iter_ft**: maximum iteration number for aggre/ ft stage.
- **align_aggre_iter_step**: step for learning rate decreasing for aggre stage.
- **ft_iter_step**: step for learning rate decreasing for ft stage.

### Unique hyper-parameters for zero-shot

- **scale_aug**: scale augmentation, it's only used in zero-shot learning. If we set it as True, the dataloader will downsample the LLR-LR pair for more data.
- **vdsr_model_path**: model path for pre-trained vdsr model.