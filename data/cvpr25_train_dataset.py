import os
import random
import math
import warnings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nibabel as nib
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import traceback
from tqdm import tqdm
from pathlib import Path
from einops import rearrange, repeat, reduce
from typing import Union, Tuple, List

from train.dist import is_master
from data.nnunet_dataset import nnUNetDatasetBlosc2
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from data.augmentation import get_training_transforms, get_validation_transforms
from data.data_loader_cvpr2025challenge import NAME2LOADER

from data.default_resampling import resample_data_or_seg, compute_new_shape

def respace_image(image: np.ndarray, current_spacing: np.ndarray, target_spacing: np.ndarray) -> np.ndarray:
    """
    Resample image to target spacing using linear interpolation.
    """
    # Add channel dimension (assuming single channel)
    if image.ndim == 3:
        image = image[np.newaxis]
        add_channel = True
    else:
        add_channel = False
    
    # Compute new shape based on spacing
    new_shape = compute_new_shape(image.shape[1:], current_spacing, target_spacing)
    
    # Resample using linear interpolation (order=1)
    resampled_image = resample_data_or_seg(
        image, 
        new_shape=new_shape,
        is_seg=False,
        axis=None,
        order=1,
        do_separate_z=False
    )
    
    # Remove channel dimension if input didn't have one
    if add_channel:
        resampled_image = resampled_image[0]
        
    return resampled_image

def respace_mask(mask: np.ndarray, current_spacing: np.ndarray, target_spacing: np.ndarray) -> np.ndarray:
    """
    Resample mask to target spacing using nearest-neighbor interpolation.
    """
    # Add channel dimension (assuming single channel)
    if mask.ndim == 3:
        mask = mask[np.newaxis]
        add_channel = True
    else:
        add_channel = False
    
    # Compute new shape based on spacing
    new_shape = compute_new_shape(mask.shape[1:], current_spacing, target_spacing)
    
    # Resample using nearest-neighbor interpolation (order=0 for seg)
    resampled_mask = resample_data_or_seg(
        mask, 
        new_shape=new_shape,
        is_seg=True,
        axis=None,
        order=0,
        do_separate_z=False
    )
    
    # Remove channel dimension if input didn't have one
    if add_channel:
        resampled_mask = resampled_mask[0]
        
    return resampled_mask

def save_predictions_to_nifti(
    mask: torch.Tensor,
    sc_pred: torch.Tensor,
    mc_pred: torch.Tensor,
    output_dir: str,
    prefix: str = "pred",
    sample_id: str = "random"
) -> None:
    """
    Save mask, single-channel prediction, and multi-channel prediction as separate NIfTI (.nii.gz) files.

    Args:
        mask (torch.Tensor): Input mask of shape (n, h, w, d), binary (0 or 1).
        sc_pred (torch.Tensor): Single-channel prediction of shape (1, h, w, d), binary.
        mc_pred (torch.Tensor): Multi-channel prediction of shape (n, h, w, d), binary.
        output_dir (str): Base directory to save NIfTI files.
        prefix (str): Prefix for output filenames (default: 'pred').
        sample_id (str): Sample identifier for subfolder; if 'random', generate a random ID (default: 'random').

    Saves:
        - output_dir/sample_{sample_id}/sc_pred/{prefix}_sc_pred.nii.gz
        - output_dir/sample_{sample_id}/mc_pred/{prefix}_mc_pred_channel_0.nii.gz, ...
        - output_dir/sample_{sample_id}/mask/{prefix}_mask_channel_0.nii.gz, ...
    """
    # Generate random sample_id if specified
    if sample_id == "random":
        sample_id = str(random.randint(1000, 9999))  # Simple 4-digit random ID
        # Alternative: sample_id = str(uuid.uuid4())[:8] for UUID-based ID

    # Append sample_id to output_dir
    sample_outdir = os.path.join(output_dir, f"sample_{sample_id}")

    # Validate inputs
    assert torch.all((mask == 0) | (mask == 1)), "Mask must be binary"
    assert torch.all((sc_pred == 0) | (sc_pred == 1)), "SC prediction must be binary"
    assert torch.all((mc_pred == 0) | (mc_pred == 1)), "MC prediction must be binary"
    n, h, w, d = mask.shape
    assert sc_pred.shape == (1, h, w, d), "SC prediction shape mismatch"
    assert mc_pred.shape == (n, h, w, d), "MC prediction shape mismatch"

    # Create subdirectories
    sc_pred_dir = os.path.join(sample_outdir, "sc_pred")
    mc_pred_dir = os.path.join(sample_outdir, "mc_pred")
    mask_dir = os.path.join(sample_outdir, "mask")
    os.makedirs(sc_pred_dir, exist_ok=True)
    os.makedirs(mc_pred_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Convert tensors to numpy
    mask_np = mask.cpu().numpy().astype(np.uint8)      # (n, h, w, d)
    sc_pred_np = sc_pred.squeeze(0).cpu().numpy().astype(np.uint8)  # (h, w, d)
    mc_pred_np = mc_pred.cpu().numpy().astype(np.uint8)  # (n, h, w, d)

    # Define affine matrix (identity, assuming spacing 1.0x1.0x1.0)
    affine = np.eye(4)

    # Save single-channel prediction
    sc_pred_nii = nib.Nifti1Image(sc_pred_np, affine)
    nib.save(sc_pred_nii, os.path.join(sc_pred_dir, f"{prefix}_sc_pred.nii.gz"))

    # Save each channel of multi-channel prediction
    for i in range(n):
        mc_pred_channel = mc_pred_np[i]  # (h, w, d)
        mc_pred_nii = nib.Nifti1Image(mc_pred_channel, affine)
        nib.save(mc_pred_nii, os.path.join(mc_pred_dir, f"{prefix}_mc_pred_channel_{i}.nii.gz"))

    # Save each channel of mask
    for i in range(n):
        mask_channel = mask_np[i]  # (h, w, d)
        mask_nii = nib.Nifti1Image(mask_channel, affine)
        nib.save(mask_nii, os.path.join(mask_dir, f"{prefix}_mask_channel_{i}.nii.gz"))

def simulate_lowres_pred(
    mask: torch.Tensor,
    drop_prob: list = [0.3, 0.9],
    add_prob: list = [0.0, 0.02], 
    channel_zero_prob: float = 0.3,
    zero_prob: float = 0.35,
    block_sizes: list = [[4, 4, 4], [8, 8, 8], [16, 16, 16], [32, 32, 32]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient CPU implementation to generate:
    1. simulated_lowres_sc_pred (1, h, w, d): Single-channel superclass prediction with drop/add.
    2. simulated_lowres_mc_pred (n, h, w, d): Multi-channel prediction with channel-specific drop/add.
    
    Args:
        mask (torch.Tensor): Input mask of shape (n, h, w, d), where each channel has values 0 or 1.
        drop_prob (list): Range of probabilities for dropping a block [min, max] (false negative).
        add_prob (list): Range of probabilities for adding a block [min, max] (false positive).
        channel_zero_prob (float): Probability of zeroing out a channel.
        zero_prob (float): Probability of returning completely zero masks.
        block_sizes (list): List of block sizes to randomly choose from.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - simulated_lowres_sc_pred (1, h, w, d), binary
            - simulated_lowres_mc_pred (n, h, w, d), binary
    """
    n, h, w, d = mask.shape

    # Early return zero masks
    if random.random() < zero_prob:
        return torch.zeros((1, h, w, d)), torch.zeros((n, h, w, d))

    # Generate random mask for channels (1=keep, 0=zero)
    channel_mask = (torch.rand(n) > channel_zero_prob).float()
    mask = mask * channel_mask.view(-1, 1, 1, 1)  # (n, h, w, d)

    # Initialize predictions
    simulated_lowres_mc_pred = mask.clone()  # (n, h, w, d)
    simulated_lowres_sc_pred = (mask.sum(dim=0, keepdim=True) > 0).float()  # (1, h, w, d)

    # Apply block-wise DROP and ADD
    if drop_prob[1] > 0 or add_prob[1] > 0:
        # Randomly select block size
        block_h, block_w, block_d = random.choice(block_sizes)

        # Compute number of blocks
        n_blocks_h = (h + block_h - 1) // block_h
        n_blocks_w = (w + block_w - 1) // block_w
        n_blocks_d = (d + block_d - 1) // block_d

        # Sample probabilities
        curr_drop_prob = random.uniform(drop_prob[0], drop_prob[1])
        curr_add_prob = random.uniform(add_prob[0], add_prob[1])

        # Generate block masks for DROP and ADD
        block_drop = torch.rand(n_blocks_h, n_blocks_w, n_blocks_d) < curr_drop_prob  # (H_blocks, W_blocks, D_blocks)
        block_add = torch.rand(n_blocks_h, n_blocks_w, n_blocks_d) < curr_add_prob    # (H_blocks, W_blocks, D_blocks)
        block_add = block_add & (~block_drop)  # Ensure no overlap

        # Upsample block masks to full resolution
        def upsample_block_mask(block_mask):
            return (
                block_mask.unsqueeze(0)  # (1, H_blocks, W_blocks, D_blocks)
                .repeat_interleave(block_h, dim=1)
                .repeat_interleave(block_w, dim=2)
                .repeat_interleave(block_d, dim=3)
                [:, :h, :w, :d]  # Crop to original size
            ).float()

        # Apply DROP and ADD to single-channel prediction
        block_drop_upsampled = upsample_block_mask(block_drop)  # (1, h, w, d)
        block_add_upsampled = upsample_block_mask(block_add)    # (1, h, w, d)
        simulated_lowres_sc_pred = simulated_lowres_sc_pred * (1 - block_drop_upsampled)
        simulated_lowres_sc_pred = torch.where(
            (simulated_lowres_sc_pred + block_add_upsampled) > 0,
            torch.ones_like(simulated_lowres_sc_pred),
            simulated_lowres_sc_pred
        )

        # Block-level channel assignment
        def assign_blocks_to_channels(block_mask):
            # Initialize channel block masks
            channel_block_mask = torch.zeros(n, n_blocks_h, n_blocks_w, n_blocks_d, dtype=torch.bool)
            # Get indices of active blocks
            active_blocks = torch.where(block_mask.flatten())[0]
            total_blocks = active_blocks.numel()

            if total_blocks > 0:
                # Randomly decide number of channels per block (at least 1)
                num_channels_per_block = torch.randint(1, n + 1, (total_blocks,), dtype=torch.int32)
                # Generate random channel indices for each block
                for idx, (block_idx, num_ch) in enumerate(zip(active_blocks, num_channels_per_block)):
                    # Convert flat index to 3D block coordinates
                    i = block_idx // (n_blocks_w * n_blocks_d)
                    j = (block_idx // n_blocks_d) % n_blocks_w
                    k = block_idx % n_blocks_d
                    # Randomly select channels
                    selected_channels = torch.randperm(n)[:num_ch]
                    channel_block_mask[selected_channels, i, j, k] = True

            # Upsample to full resolution
            channel_mask_upsampled = channel_block_mask.view(n, n_blocks_h, n_blocks_w, n_blocks_d)
            channel_mask_upsampled = (
                channel_mask_upsampled
                .unsqueeze(0)  # (1, n, H_blocks, W_blocks, D_blocks)
                .repeat_interleave(block_h, dim=2)
                .repeat_interleave(block_w, dim=3)
                .repeat_interleave(block_d, dim=4)
                [:, :, :h, :w, :d]  # Crop to (1, n, h, w, d)
            ).squeeze(0)  # (n, h, w, d)
            return channel_mask_upsampled

        # Assign drop and add blocks to channels
        channel_keep_mask = assign_blocks_to_channels(~block_drop)  # (n, h, w, d)
        channel_add_mask = assign_blocks_to_channels(block_add)    # (n, h, w, d)

        # Update multi-channel prediction
        simulated_lowres_mc_pred = torch.where(
            channel_keep_mask,
            simulated_lowres_mc_pred,  
            torch.zeros_like(simulated_lowres_mc_pred) 
        )
        simulated_lowres_mc_pred = torch.where(
            channel_add_mask,
            torch.ones_like(simulated_lowres_mc_pred),
            simulated_lowres_mc_pred
        )

    # Ensure binary output
    simulated_lowres_sc_pred = (simulated_lowres_sc_pred > 0).float()
    simulated_lowres_mc_pred = (simulated_lowres_mc_pred > 0).float()

    return simulated_lowres_sc_pred, simulated_lowres_mc_pred
    
def contains(text, key):
    if isinstance(key, str):
        return key in text
    elif isinstance(key, list):
        for k in key:
            if k in text:
                return True
        return False         
    
class Med_SAM_Dataset(Dataset):
    def __init__(self,
                 jsonl_file, 
                 text_prompts_json,
                 dataset_config,
                 batch_size=2,
                 crop_size=[288, 288, 96],
                 target_spacing=[1.0, 1.0, 1.0],
                 max_queries=16,
                 allow_repeat=True,
                 nnUNet_aug=True,
                 deep_supervision=True):
        """
        Assemble segmentation datasets
        
        Args:
            jsonl_file (_type_): a jsonl contains all train sample information
            crop_size (int, optional): _description_. Defaults to [288,288,96].
            max_queries (int, optional): _description_. Defaults to 32.
            # dataset_config (str, optional): a path to config file, defining the sampling, loading parameters of each dataset etc
            allow_repeat (bool, optional): sample for multiply times to accelerate convergency. Defaults to True.
        """
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.max_queries = max_queries
        self.target_spacing = target_spacing  # target spacing for resampling

        self.oversample_foreground_percent = 0.2 #0.33
        probabilistic_oversampling = True
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        
        # load data configs
        with open(dataset_config, 'r') as f:
            self.dataset_config = json.load(f)

        # self.deep_supervision = deep_supervision
        # self.pool_op_kernel_sizes = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

        # deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_crop_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_crop_size()
        print("mirror_axes: ", mirror_axes)
        self.initial_crop_size = initial_crop_size

        if len(initial_crop_size) == 2:
            crop_size = (1, *initial_crop_size)
            crop_size = (1, *initial_crop_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False
        
        pad_sides = None
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(initial_crop_size) - np.array(crop_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        
        # load samples
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        
        # load intensity 2 label json
        with open(text_prompts_json, 'r') as f:
            self.text_prompts_json = json.load(f)
        
        # statistics the size of each dataset and the repeated times and the sampled times within a log interval
        datasets_dist = [l['dataset'] for l in lines]
        self.datasets = set(datasets_dist)
        self.datasets_size = {}
        self.datasets_repeat_times = {}
        for dataset in self.datasets:
            self.datasets_size[dataset] = datasets_dist.count(dataset)
            self.datasets_repeat_times[dataset] = 0       
    
        self.data = []   # list of data samples
        self.sample_weight = []  # and their sampling weight
        count_repeat = 0
        
        for sample in lines:
            
            # sampling weight : inverse to square root of dataset size
            size = self.datasets_size[sample['dataset']]
            weight = 1 / (math.sqrt(size))
            # sampling weight : allow manual adjustment in data config file
            weight = weight * self.dataset_config[sample['dataset']]['sampling_weight']
            
            # repeat times for label num
            label_num = len(self.text_prompts_json[sample['dataset']]) - 1 # exclude instance label
            query_repeat_times = max(1, (label_num / max_queries))
            # repeat times for roi size
            if 'roi_y1x1z1_y2x2z2' in sample and sample['roi_y1x1z1_y2x2z2']:
                y1, x1, z1, y2, x2, z2 = sample['roi_y1x1z1_y2x2z2']
                h_repeat_times = max(1, ((y2-y1) / crop_size[0]))
                w_repeat_times = max(1, ((x2-x1) / crop_size[1]))
                d_repeat_times = max(1, ((z2-z1) / crop_size[2]))
                size_repeat_times = h_repeat_times * w_repeat_times * d_repeat_times
            else:
                size_repeat_times = 1
                
            # not repeat
            if not allow_repeat:
                size_repeat_times = query_repeat_times = 1
                
            # allow repeat
            repeat_times = round(size_repeat_times * query_repeat_times)  # e.g. 1.5 * 2.5 = 3.75 --> 4
            for i in range(round(repeat_times)):
                self.data.append(sample)
                self.sample_weight.append(weight)
            count_repeat += (repeat_times - 1)
            self.datasets_repeat_times[sample['dataset']] += (repeat_times - 1)
            
        """
        # determine sample weight and num
        self.num_2d = 0
        self.num = len(self.data)
        self.data_split = {'2d':[0, self.num_2d], '3d':[self.num_2d, self.num_2d+self.num]}
        """
        
        if is_master():
            print(f'** DATASET ** {len(lines)} unique 3D samples are loaded, {count_repeat} samples are repeated')  
            print(f'** DATASET ** In total {len(self.datasets)} datasets.\n')
            print(f'** DATASET ** Size, Repeated Times and Repeat/Size Ratio for each dataset:\n')
            for k,repeated_times in self.datasets_repeat_times.items():
                size = self.datasets_size[k]
                print(f'{k} : {size}/{repeated_times} = {repeated_times/size}')
        
        # data augmentation (tailor for each dataset)
        self.nnUNet_aug = nnUNet_aug
        if nnUNet_aug:
            # training pipeline
            self.augmentator = get_training_transforms(self.datasets,
                crop_size, rotation_for_DA, None, mirror_axes, do_dummy_2d_data_aug,
                order_resampling_data=3, order_resampling_seg=1,
                use_mask_for_norm=None,
                is_cascaded=False, foreground_labels=None,
                regions=None,
                ignore_label=None)
        else:
            print("error: nnUNet augmentation is not supported yet")
            exit()
            # self.augmentator = get_SAT_augmentator(self.dataset_config, self.datasets)
    
    def __len__(self):
        # DEBUG
        # return len(self.data)
        return 1000000000 # life long training ... (10e9)

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent
    
    @staticmethod
    def get_crop_size(final_crop_size, rot_x, rot_y, rot_z, scale_range):
        if isinstance(rot_x, (tuple, list)):
            rot_x = max(np.abs(rot_x))
        if isinstance(rot_y, (tuple, list)):
            rot_y = max(np.abs(rot_y))
        if isinstance(rot_z, (tuple, list)):
            rot_z = max(np.abs(rot_z))
        rot_x = min(90 / 360 * 2. * np.pi, rot_x)
        rot_y = min(90 / 360 * 2. * np.pi, rot_y)
        rot_z = min(90 / 360 * 2. * np.pi, rot_z)
        from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
        coords = np.array(final_crop_size)
        final_shape = np.copy(coords)
        if len(coords) == 3:
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
        elif len(coords) == 2:
            final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
        final_shape /= min(scale_range)
        return final_shape.astype(int)

    def configure_rotation_dummyDA_mirroring_and_inital_crop_size(self):
        ANISO_THRESHOLD = 3
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        crop_size = self.crop_size
        dim = len(crop_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(crop_size) / min(crop_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have crop_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(crop_size) / crop_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_crop_size = self.get_crop_size(crop_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_crop_size[0] = crop_size[0]

        print(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_crop_size, mirror_axes

    def _merge_modality(self, mod):
        if contains(mod, ['mr', 't1', 't2', 'mri', 'flair', 'dwi']):
            return 'mri'
        if contains(mod, 'ct'):
            return 'ct'
        if contains(mod, 'pet'):
            return 'pet'
        if contains(mod, ['us', 'us3d', 'ultrasound']):
            return 'us'
        if contains(mod, ['microscopy']):
            return 'microscopy'
        else:
            raise ValueError(f'Unknown modality {mod}')
    
    def _pad_if_necessary(self, image=None, mask=None):
        # image size >= crop size 
        if not (image is None):
            c, h, w, d = image.shape
            croph, cropw, cropd = self.crop_size
            pad_in_h = 0 if h >= croph else croph - h
            pad_in_w = 0 if w >= cropw else cropw - w
            pad_in_d = 0 if d >= cropd else cropd - d
            if pad_in_h + pad_in_w + pad_in_d > 0:
                pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
                image = F.pad(image, pad, 'constant', 0)   # chwd
        
        if not (mask is None):
            n, h, w, d = mask.shape
            croph, cropw, cropd = self.crop_size
            pad_in_h = 0 if h >= croph else croph - h
            pad_in_w = 0 if w >= cropw else cropw - w
            pad_in_d = 0 if d >= cropd else cropd - d
            if pad_in_h + pad_in_w + pad_in_d > 0:
                pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
                mask = F.pad(mask, pad, 'constant', 0)   # nhwd
        
        return image, mask
    
    def _crop(self, image, mc_mask, is_roi_crop, label_based_crop_prob, uncenter_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        if (imgh - croph) > 0 or (imgw - cropw) > 0 or (imgd - cropd) > 0:
            # need crop
            if (not mc_mask.any()) or (not is_roi_crop):
                # no roi region
                image, y1x1z1_y2x2z2 = self._random_crop(image)
            else:
                # 100% roi crop
                image, y1x1z1_y2x2z2 = self._roi_crop(image, mc_mask, label_based_crop_prob, uncenter_prob)
        else:
            y1x1z1_y2x2z2 = [0, 0, 0, imgh, imgw, imgd]
                
        return image, y1x1z1_y2x2z2
    
    def _roi_crop(self, image, mc_mask, label_based_crop_prob, uncenter_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        
        if random.random() < label_based_crop_prob:
            # find a pos label and crop based on it (ensure at least one pos label before roi crop
            pos_label_idx_ls = [i for i in range(mc_mask.shape[0]) if mc_mask[i].any()]
            pos_label_idx = random.sample(pos_label_idx_ls, 1)[0]
            mask_to_select = mc_mask[pos_label_idx, :, :, :]  # h w d 
        else:
            # crop based on all labels
            mask_to_select = mc_mask.any(dim=0)
        
        # select a voxel
        voxels_foreground = torch.nonzero(mask_to_select, as_tuple=True)
        selected_index = random.randint(0, len(voxels_foreground[0])-1)
        selected_voxel = (voxels_foreground[0][selected_index].item(), voxels_foreground[1][selected_index].item(), voxels_foreground[2][selected_index].item())
        
        # check the boundary
        if selected_voxel[0] - croph // 2 > 0:
            start_y = selected_voxel[0] - croph // 2
            if start_y + croph < imgh:
                end_y = start_y + croph
            else:
                end_y = imgh
                start_y = imgh-croph
        else:
            start_y = 0
            end_y = croph
            
        if selected_voxel[1] - cropw // 2 > 0:
            start_x = selected_voxel[1] - cropw // 2
            if start_x + cropw < imgw:
                end_x = start_x + cropw
            else:
                end_x = imgw
                start_x = imgw-cropw
        else:
            start_x = 0
            end_x = cropw

        if selected_voxel[2] - cropd // 2 > 0:
            start_z = selected_voxel[2] - cropd // 2
            if start_z + cropd < imgd:
                end_z = start_z + cropd
            else:
                end_z = imgd
                start_z = imgd-cropd
        else:
            start_z = 0
            end_z = cropd  
        
        # randomly shift the crop (must contain the selected voxel
        if random.random() < uncenter_prob:
            y_left_space = min(start_y - 0, end_y - selected_voxel[0])
            y_right_space = min(imgh - end_y, selected_voxel[0] - start_y)
            y_adjust = random.randint(-1 * y_left_space, y_right_space)
            start_y += y_adjust
            end_y += y_adjust
            
            x_left_space  = min(start_x-0, end_x-selected_voxel[1])
            x_right_space = min(imgw-end_x, selected_voxel[1]-start_x)
            x_adjust = random.randint(-1*x_left_space, x_right_space)
            start_x += x_adjust
            end_x += x_adjust

            z_left_space = min(start_z - 0, end_z - selected_voxel[2])
            z_right_space = min(imgd - end_z, selected_voxel[2] - start_z)
            z_adjust = random.randint(-1 * z_left_space, z_right_space)
            start_z += z_adjust
            end_z += z_adjust
            
        # crop
        crop_image = image[:, start_y:end_y, start_x:end_x, start_z:end_z]

        return crop_image, [start_y, start_x, start_z, end_y, end_x, end_z]
    
    def _random_crop(self, image):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        # 
        start_y = random.randint(0, imgh - croph)
        end_y = start_y + croph
        start_x = random.randint(0, imgw - cropw)
        end_x = start_x + cropw
        start_z = random.randint(0, imgd - cropd)
        end_z = start_z + cropd
        #
        crop_image = image[:, start_y:end_y, start_x:end_x, start_z:end_z]
        
        return crop_image, [start_y, start_x, start_z, end_y, end_x, end_z]
    
    def _select_pos_labels(self, label_index_ls, is_pos_ls, neg_label_ratio_threshold):
        """        
        Args:
            label_index_ls (List of int) : candidate labels (channel index in segmentation mask)
            is_pos_ls (List of bool) : positive label (True) or not (False), equal length to label_index_ls
        
        Returns:
            chosen_label_index_ls (List of int) : chosen subset of label_index_ls
            chosen_is_pos (List of bool) : chosen subset of is_pos_ls
        """
        # divide all the labels into pos and neg
        pos_label_index_ls = []
        neg_label_index_ls = []
        for i, is_pos in zip(label_index_ls, is_pos_ls):
            if is_pos:
                pos_label_index_ls.append(i)
            else:
                neg_label_index_ls.append(i)
        pos_num = len(pos_label_index_ls)
        neg_num = len(neg_label_index_ls)
        
        if pos_num == 0:
            # degrad to random sample
            sample_num = min(self.max_queries, len(label_index_ls))
            chosen_label_index_ls = random.sample(label_index_ls, sample_num)
            chosen_is_pos = [False] * sample_num
            return chosen_label_index_ls, chosen_is_pos
        
        # indicate each sample is pos or neg
        chosen_is_pos = []
        
        if pos_num <= self.max_queries:
            # all pos labels are included, then sample some neg labels
            chosen_label_index_ls = pos_label_index_ls 
            chosen_is_pos += [True] * pos_num
            max_neg_num = int(neg_label_ratio_threshold * pos_num)    # neg label num < (pos label num) * x%
            left_pos_num = min(self.max_queries-pos_num, max_neg_num)   # neg label num < self.max_queries-pos_num
            if neg_num <= left_pos_num:
                # neg are all sampled
                chosen_label_index_ls += neg_label_index_ls
                chosen_is_pos += [False] * neg_num
            else:
                # neg are sampled to control the ratio and max label num
                chosen_label_index_ls += random.sample(neg_label_index_ls, left_pos_num)
                chosen_is_pos += [False] * left_pos_num
        else:
            # no neg labels are sampled
            chosen_label_index_ls = random.sample(pos_label_index_ls, self.max_queries)
            chosen_is_pos += [True] * self.max_queries

        return chosen_label_index_ls, chosen_is_pos
    
    def is_overlap(self, a_y1x1z1_y2x2z2, b_y1x1z1_y2x2z2):
        # judge is overlap or not between two cubes
        a_y1, a_x1, a_z1, a_y2, a_x2, a_z2 = a_y1x1z1_y2x2z2
        b_y1, b_x1, b_z1, b_y2, b_x2, b_z2 = b_y1x1z1_y2x2z2
        overlap_x = not (a_x2 < b_x1 or b_x2 < a_x1)
        overlap_y = not (a_y2 < b_y1 or b_y2 < a_y1)
        overlap_z = not (a_z2 < b_z1 or b_z2 < a_z1)
        return overlap_x and overlap_y and overlap_z
    
    def _find_pos_labels_in_crop(self, crop_y1x1z1_y2x2z2, labels_y1x1z1_y2x2z2):
        is_pos = []
        for y1x1z1_y2x2z2 in labels_y1x1z1_y2x2z2:
            if y1x1z1_y2x2z2 and self.is_overlap(y1x1z1_y2x2z2, crop_y1x1z1_y2x2z2):
                is_pos.append(True)
            else:
                is_pos.append(False)
        return is_pos
    
    def get_size_and_repeat(self, dataset_name):
        return self.datasets_size[dataset_name], self.datasets_repeat_times[dataset_name]
    
    def sc_mask_to_mc_mask(self, sc_mask, label_values_ls):
        assert sc_mask.ndim == 3
        h, w, d = sc_mask.shape
        n = len(label_values_ls)
        mc_mask = torch.zeros((n, h, w, d), dtype=bool)
        for i, label_value in enumerate(label_values_ls):
            mc_mask[i] = torch.where(sc_mask == label_value, 1, 0)
        return mc_mask

    def select_text_prompts(self, lists):
        """
        Select text prompts
        
        Args:
            lists (List): lists of text prompt strings

        Returns:
            selected N elements, and which label they from
        """
        
        queues = []
        for orig_idx, lst in enumerate(lists):
            if lst:
                shuffled = lst.copy()
                random.shuffle(shuffled)
                queues.append((orig_idx, shuffled))
        
        random.shuffle(queues)
        
        collected_elements = []
        source_indices = []
        
        while len(collected_elements) < self.max_queries and queues:
            progress = False
            
            for i in reversed(range(len(queues))):
                if len(collected_elements) >= self.max_queries:
                    break
                orig_idx, elements = queues[i]
                if elements:
                    element = elements.pop()
                    collected_elements.append(element)
                    source_indices.append(orig_idx)
                    progress = True
                    if not elements:
                        queues.pop(i)
            
            if not progress:
                break
        
        return collected_elements[:self.max_queries], source_indices[:self.max_queries]
    
    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None], label_values_ls: list,
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        
        label_values_ls = list(np.unique(label_values_ls))
        label_values_ls.sort()
        self.annotated_classes_key = tuple([-1] + label_values_ls)

        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.initial_crop_size[d]:
                need_to_pad[d] = self.initial_crop_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - initial_crop_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.initial_crop_size[i] for i in range(dim)]

        self.has_ignore = False

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    warnings.warn('Warning! No annotated pixels in image!')
                    selected_class = None
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')

            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.initial_crop_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.initial_crop_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs
    
    def load_data(self, dataset_name, data_path):

        label_2_text_prompt = self.text_prompts_json[dataset_name] # '1':['xxx', 'xxx', ...]
        label_values_ls = list(label_2_text_prompt.keys())
        label_values_ls = [int(v) for v in label_values_ls if v!='instance_label']
        text_prompt_ls = list(label_2_text_prompt.values()) # list of list of str
        text_prompt_ls = [ls for ls in text_prompt_ls if isinstance(ls, list)]

        x, y, z = self.target_spacing
        ps_x, ps_y, ps_z = self.crop_size
        target_spacing_folder_name = "3D_train_npz_all_spacing_xX_yX_zX_ps_A_B_C".replace('xX', f'x{x}').replace('yX', f'y{y}').replace('zX', f'z{z}').replace('A', f'{ps_x}').replace('B', f'{ps_y}').replace('C', f'{ps_z}')

        # load respacing data
        data_path = data_path.replace('CVPR-BiomedSegFM/3D_train_npz_all', 'CVPR-BiomedSegFM_preprocess/' + target_spacing_folder_name)

        data_folder = os.path.dirname(data_path)

        blosc2_ds = nnUNetDatasetBlosc2(data_folder)
        identifier = os.path.basename(data_path)[:-4]

        j = np.random.randint(0, self.batch_size)
        force_fg = self.get_do_oversample(j)
        data, seg, seg_prev, properties = blosc2_ds.load_case(identifier)

        shape = data.shape[1:]

        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'], label_values_ls)
        bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

        # use ACVL utils for that. Cleaner.
        img = crop_and_pad_nd(data, bbox, 0)

        seg_cropped = crop_and_pad_nd(seg, bbox, -1)
        if seg_prev is not None:
            seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
        
        sc_mask = seg_cropped
        
        return img, sc_mask, text_prompt_ls, label_values_ls
        
    def __getitem__(self, idx):
        while True:
            try: 
                # DEBUG
                # sample = self.data[idx]
                sample = random.choices(self.data, weights=self.sample_weight)[0]
                
                dataset_name = sample['dataset']
                data_path = sample['data']

                image, mask, text_prompt_ls, label_values_ls = self.load_data(dataset_name, data_path)
                    
                modality = sample['data'].split('/')[-3]
                modality = self._merge_modality(modality.lower())   

                image = image[np.newaxis, ...]  # 1 1 h w d
                mask = mask[np.newaxis, ...]  # 1 1 h w d
                data_dict = {'data': image, 'seg': mask}
                data_dict = self.augmentator[dataset_name](**data_dict)
                image, mask = data_dict['data'], data_dict['target']
                image = image[0, ...]
                mask = mask[0, ...]

                mask = mask.squeeze(0)  # h w d
                mask = self.sc_mask_to_mc_mask(mask, label_values_ls)  # 1 h w d --> n h w d

                # for all the label in this sample, check if positive in the cropped patch
                is_pos_in_crop = [mask[i].any().item() for i in range(mask.shape[0])]
                # sample from all the labels based on the cropped patch (to balance pos and neg labels)
                neg_label_ratio_threshold = 0.33 #0.2
                all_label_index_ls = [i for i in range(len(is_pos_in_crop))]
                chosen_label_index_ls, is_pos_ls = self._select_pos_labels(all_label_index_ls, is_pos_in_crop, neg_label_ratio_threshold)   # [label1, label2, ....], [True, False, ...]

                # so these are the chosen labels and their mask
                chosen_label = [text_prompt_ls[i] for i in chosen_label_index_ls]
                mask = mask[chosen_label_index_ls]

                if len(chosen_label) == self.max_queries:
                    chosen_label = [random.choice(ls) for ls in chosen_label]
                else:
                    chosen_label, source_indices = self.select_text_prompts(chosen_label)
                    indices = np.array(source_indices)
                    mask = mask[indices]
                    
                if not isinstance(mask, torch.FloatTensor):
                    mask = mask.float()
                
                simulated_lowres_sc_pred, simulated_lowres_mc_pred = simulate_lowres_pred(
                                    mask,
                                    drop_prob=[0.3, 0.9],
                                    add_prob=[0.0, 0.02],
                                    channel_zero_prob=0.1,
                                    zero_prob=0.35,
                                    block_sizes=[[4, 4, 4], [8, 8, 8], [16, 16, 16], [32, 32, 32]])           

                # tmp_outdir = "/data/shipengcheng/code/CVPR2025_Text_guided_seg/log/nano_UNet_CVPR2025_crop_size_160_160_160_spacing_1.0_1.0_1.0_w_simulated_lowres_pretrain_nano_cvpr25_v0/checkpoint/tmp_save"
                # save_predictions_to_nifti(mask=mask, sc_pred=simulated_lowres_sc_pred, mc_pred=simulated_lowres_mc_pred, output_dir=tmp_outdir, prefix=dataset_name, sample_id="random")

                # simple check
                _, H, W, D = image.shape
                N, mH, mW, mD = mask.shape
                assert H == mH and W == mW and D == mD, f'image shape {H, W, D} inconsistent with mask shape {mH, mW, mD}'
                assert N == len(chosen_label), f'query num {len(chosen_label)} inconsistent with gt mask channels {N}'
                    
                break
            except SystemExit:
                exit()
            except:
                # record bugs in loading data
                traceback_info = traceback.format_exc()
                print(f'*** {dataset_name} *** {data_path} ***\n')
                print(traceback_info)

        return {'image':image, 'mask':mask, 'text':chosen_label, 'simulated_lowres_sc_pred':simulated_lowres_sc_pred, 'simulated_lowres_mc_pred':simulated_lowres_mc_pred, 'modality':modality, 'image_path':data_path, 'mask_path':data_path, 'dataset':dataset_name}
    
if __name__ == '__main__':
    dataset = Med_SAM_Dataset(
        '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/train_10percent_raw_subset.jsonl', 
        '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/SegFM3D/CVPR25_TextSegFMData_with_class.json',
        '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/data/dataset_config/cvpr25.json',
        crop_size=[288,288,96], 
        max_queries=16, 
        allow_repeat=True,
        nnUNet_aug=True
    )
    
    debug_dir = "/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/data/cvpr25_train_debug_visualization"
    os.makedirs(debug_dir, exist_ok=True)

    # For debugging, iterate over a fixed number of samples (e.g., 10)
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        image, mask = sample["image"], sample["mask"]
        data_path = sample["image_path"]
        ds_name = sample['dataset']
        text = sample['text']
        basename = os.path.splitext(os.path.basename(data_path))[0]
        sample_dir = os.path.join(debug_dir, ds_name)
        os.makedirs(sample_dir, exist_ok=True)

        affine = np.eye(4)

        # Convert tensor to numpy array; remove singleton channel dimension if necessary
        img_np = image.numpy()
        if img_np.shape[0] == 1:
            img_np = img_np[0]
        nib.save(nib.Nifti1Image(img_np, affine), os.path.join(sample_dir, f"(img){basename}.nii.gz"))

        mask_np = mask.numpy()
        results = np.zeros((mask_np.shape[1], mask_np.shape[2], mask_np.shape[3])) # hwd
        for j in range(mask_np.shape[0]):
            results += mask_np[j, :, :, :] * (j+1)   # 0 --> 1 (skip background)          
        nib.save(nib.Nifti1Image(results, affine), os.path.join(sample_dir, f"(seg){basename}.nii.gz"))
        
        with open(os.path.join(sample_dir, f"{basename}.txt"), 'w') as f:
            for i, t in enumerate(text):
                f.write(f'{i} : {t}\n')
        