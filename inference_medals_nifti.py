"""
Medal-S inference script for generic raw image segmentation.

This script provides an interface for running Medal-S inference
on raw NIfTI images. It supports both single-stage (Stage 2 only) and 
two-stage (Stage 1 + Stage 2) inference modes.

Usage:
    python inference_medals.py --input input.nii.gz --output output.nii.gz \\
        --modality CT --texts "Aorta observed in abdominal CT scans" --labels 1
    
    # Or use JSON configuration file:
    python inference_medals.py --input input.nii.gz --output output.nii.gz \\
        --config config.json --mode stage1+stage2

Author: Pengcheng Shi
Institute: Medical Image Insights, Inc., Shanghai, China
Email: shipc1220@gmail.com
License: Apache License 2.0
"""

import os
import argparse
import json
import time
import math
import random
import itertools
import gc
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from typing import List
from scipy.ndimage import label, gaussian_filter
from einops import rearrange
from tqdm import tqdm
from torch.cuda.amp import autocast

from data.default_resampling import resample_data_or_seg, compute_new_shape, resample_data_or_seg_to_spacing
from data.resample_torch import resample_torch_fornnunet, resample_torch_simple
from model.maskformer import Maskformer
from model.knowledge_encoder import Knowledge_Encoder

def adjust_spacing(img_array, img_spacing):
    """
    Adjust spacing based on image dimensions.
    
    This function swaps spacing values if the dimension with minimum size
    doesn't match the dimension with maximum spacing.
    
    Args:
        img_array: Image array (used for shape reference)
        img_spacing: Spacing array
    
    Returns:
        Adjusted spacing array
    """
    img_spacing = np.asarray(img_spacing)
    min_dim_index = np.argmin(img_array.shape)
    max_spacing_index = np.argmax(img_spacing)
    
    if (min_dim_index != max_spacing_index) and (img_spacing[max_spacing_index] > 0.5):
        new_order = list(range(len(img_spacing)))
        new_order[min_dim_index], new_order[max_spacing_index] = new_order[max_spacing_index], new_order[min_dim_index]
        img_spacing = img_spacing[new_order]
    
    return img_spacing


def remove_small_objects_binary(binary_data, min_size=10):
    """
    Remove small objects from binary data.
    
    Args:
        binary_data: Binary array
        min_size: Minimum size threshold for objects to keep
    
    Returns:
        Binary array with small objects removed
    """
    labeled_array, num_features = label(binary_data)
    sizes = np.bincount(labeled_array.ravel())
    remove = sizes < min_size
    remove[0] = False  # Ensure the background (label 0) is not removed
    labeled_array[remove[labeled_array]] = 0
    return labeled_array > 0


def respace_image(image: np.ndarray, current_spacing: np.ndarray, target_spacing: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Resample image to target spacing.
    
    Args:
        image: Input image array with shape (C, H, W, D)
        current_spacing: Current spacing array
        target_spacing: Target spacing array
        device: PyTorch device for resampling
    
    Returns:
        Resampled image array
    """
    new_shape = compute_new_shape(image.shape[1:], current_spacing, target_spacing)
    resampled_image = resample_torch_fornnunet(
        image, new_shape, current_spacing, target_spacing,
        is_seg=False, num_threads=8, device=device,
        memefficient_seg_resampling=False,
        force_separate_z=None,
        separate_z_anisotropy_threshold=3.0
    )
    return resampled_image


def respace_mask(mask: np.ndarray, current_spacing: np.ndarray, target_spacing: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Resample mask to target spacing.
    
    Args:
        mask: Input mask array with shape (C, H, W, D)
        current_spacing: Current spacing array
        target_spacing: Target spacing array
        device: PyTorch device for resampling
    
    Returns:
        Resampled mask array
    """
    new_shape = compute_new_shape(mask.shape[1:], current_spacing, target_spacing)
    resampled_mask = resample_torch_fornnunet(
        mask, new_shape, current_spacing, target_spacing,
        is_seg=True, num_threads=8, device=device,
        memefficient_seg_resampling=False,
        force_separate_z=None,
        separate_z_anisotropy_threshold=3.0
    )
    return resampled_mask


def split_3d(image_tensor, crop_size=[288, 288, 96]):
    """
    Split 3D image into overlapping patches.
    
    Patches are extracted with 50% overlap (stride = crop_size / 2) to ensure
    complete coverage of the image volume.
    
    Args:
        image_tensor: Input image tensor with shape (C, H, W, D)
        crop_size: Size of each patch [h, w, d]
    
    Returns:
        split_patch: List of patch tensors
        split_idx: List of patch indices [h_s, h_e, w_s, w_e, d_s, d_e]
    """
    interval_h, interval_w, interval_d = crop_size[0] // 2, crop_size[1] // 2, crop_size[2] // 2
    split_idx = []
    split_patch = []

    c, h, w, d = image_tensor.shape
    h_crop = max(math.ceil(h / interval_h) - 1, 1)
    w_crop = max(math.ceil(w / interval_w) - 1, 1)
    d_crop = max(math.ceil(d / interval_d) - 1, 1)

    for i in range(h_crop):
        h_s = i * interval_h
        h_e = h_s + crop_size[0]
        if h_e > h:
            h_s = h - crop_size[0]
            h_e = h
            if h_s < 0:
                h_s = 0
        for j in range(w_crop):
            w_s = j * interval_w
            w_e = w_s + crop_size[1]
            if w_e > w:
                w_s = w - crop_size[1]
                w_e = w
                if w_s < 0:
                    w_s = 0
            for k in range(d_crop):
                d_s = k * interval_d
                d_e = d_s + crop_size[2]
                if d_e > d:
                    d_s = d - crop_size[2]
                    d_e = d
                if d_s < 0:
                    d_s = 0
                split_idx.append([h_s, h_e, w_s, w_e, d_s, d_e])
                split_patch.append(image_tensor[:, h_s:h_e, w_s:w_e, d_s:d_e])
                
    return split_patch, split_idx


def pad_if_necessary(image, crop_size=[288, 288, 96]):
    """
    Pad image if necessary to meet crop size requirements.
    
    Args:
        image: Input image tensor with shape (C, H, W, D)
        crop_size: Minimum size requirements [h, w, d]
    
    Returns:
        padded_image: Padded image tensor
        padding_info: Tuple of padding amounts (pad_h, pad_w, pad_d)
    """
    c, h, w, d = image.shape
    croph, cropw, cropd = crop_size
    pad_in_h = 0 if h >= croph else croph - h
    pad_in_w = 0 if w >= cropw else cropw - w
    pad_in_d = 0 if d >= cropd else cropd - d
    
    padding_info = (pad_in_h, pad_in_w, pad_in_d)
    
    if pad_in_h + pad_in_w + pad_in_d > 0:
        pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
        image = F.pad(image, pad, 'constant', 0)
    
    return image, padding_info


def remove_padding(padded_image, padding_info):
    """
    Remove padding from image.
    
    Args:
        padded_image: Padded image (can be torch.Tensor or numpy array)
        padding_info: Tuple of padding amounts (pad_h, pad_w, pad_d)
    
    Returns:
        Image with padding removed
    """
    pad_in_h, pad_in_w, pad_in_d = padding_info
    
    if len(padded_image.shape) == 4:
        if isinstance(padded_image, torch.Tensor):
            return padded_image[:, :padded_image.shape[1]-pad_in_h, :padded_image.shape[2]-pad_in_w, :padded_image.shape[3]-pad_in_d]
        else:
            return padded_image[:, :padded_image.shape[1]-pad_in_h, :padded_image.shape[2]-pad_in_w, :padded_image.shape[3]-pad_in_d]
    else:
        if isinstance(padded_image, torch.Tensor):
            return padded_image[:padded_image.shape[0]-pad_in_h, :padded_image.shape[1]-pad_in_w, :padded_image.shape[2]-pad_in_d]
        else:
            return padded_image[:padded_image.shape[0]-pad_in_h, :padded_image.shape[1]-pad_in_w, :padded_image.shape[2]-pad_in_d]


def internal_maybe_mirror_and_predict(model=None, queries=None, image_input=None, simulated_lowres_sc_pred=None, 
                                      simulated_lowres_mc_pred=None, mirror_axes=(0, 1, 2)):
    """
    Apply test-time augmentation with mirroring.
    
    This function performs inference with multiple mirroring combinations
    and averages the results for improved robustness.
    
    Args:
        model: Model to use for prediction
        queries: Query tensor
        image_input: Input image tensor
        simulated_lowres_sc_pred: Simulated low-res single-channel prediction
        simulated_lowres_mc_pred: Simulated low-res multi-channel prediction
        mirror_axes: Axes to mirror (0, 1, 2 for spatial dimensions)
    
    Returns:
        Averaged prediction tensor
    """
    prediction = model(queries=queries, 
                       image_input=image_input, 
                       simulated_lowres_sc_pred=simulated_lowres_sc_pred, 
                       simulated_lowres_mc_pred=simulated_lowres_mc_pred, 
                       train_mode=False)

    if mirror_axes is not None:
        assert max(mirror_axes) <= image_input.ndim - 3, 'mirror_axes does not match the dimension of the input!'
        mirror_axes = [m + 2 for m in mirror_axes]
        axes_combinations = [
            c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
        ]
        for axes in axes_combinations:
            image_input_fliped = torch.flip(image_input, axes)
            simulated_lowres_sc_pred_fliped = torch.flip(simulated_lowres_sc_pred.unsqueeze(0), axes).squeeze(0) if simulated_lowres_sc_pred is not None else None
            simulated_lowres_mc_pred_fliped = torch.flip(simulated_lowres_mc_pred.unsqueeze(0), axes).squeeze(0) if simulated_lowres_mc_pred is not None else None
            prediction_fliped = model(queries=queries, 
                                     image_input=image_input_fliped, 
                                     simulated_lowres_sc_pred=simulated_lowres_sc_pred_fliped, 
                                     simulated_lowres_mc_pred=simulated_lowres_mc_pred_fliped, 
                                     train_mode=False)
            prediction += torch.flip(prediction_fliped, axes)
        prediction /= (len(axes_combinations) + 1)
    return prediction


def compute_patch_prediction(
    queries: torch.Tensor,
    patches: torch.Tensor,
    lowres_single_channel_pred: torch.Tensor,
    lowres_multi_channel_pred: torch.Tensor,
    model: torch.nn.Module,
    possible_block_sizes: List[int],
    n_repeats: int = 1,
    disable_tta: bool = True
) -> torch.Tensor:
    """
    Compute patch predictions using complementary masking.
    
    This function splits the volume into blocks, processes complementary halves
    using random masks, and combines results. The process is repeated n_repeats
    times with different random masks, and results are averaged.
    
    Args:
        queries: Input query tensor, shape (batch, query_dim)
        patches: Image patch tensor, shape (batch, channels, h, w, d)
        lowres_single_channel_pred: Low-res single-channel prediction, shape (1, 1, h, w, d)
        lowres_multi_channel_pred: Low-res multi-channel prediction, shape (1, c, h, w, d)
        model: Trained neural network model
        possible_block_sizes: List of possible block sizes (e.g., [8, 16, 32])
        n_repeats: Number of times to repeat prediction with different masks
        disable_tta: Whether to disable test-time augmentation
    
    Returns:
        Averaged patch prediction, shape (1, c, h, w, d)
    """
    # Validate inputs
    if not possible_block_sizes:
        raise ValueError("possible_block_sizes cannot be empty")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    
    _, _, h, w, d = lowres_single_channel_pred.shape
    device = lowres_single_channel_pred.device
    prediction_sum = torch.zeros_like(lowres_multi_channel_pred, device=device)

    def upsample_block_mask(block_mask: torch.Tensor, block_size: int) -> torch.Tensor:
        """Upsample a block mask to full resolution."""
        upsampled = (
            block_mask.unsqueeze(0).unsqueeze(0)
            .repeat_interleave(block_size, dim=2)
            .repeat_interleave(block_size, dim=3)
            .repeat_interleave(block_size, dim=4)
            [:, :, :h, :w, :d]
        ).float()
        return upsampled

    for _ in range(n_repeats):
        block_size = random.choice(possible_block_sizes)
        n_blocks_h = (h + block_size - 1) // block_size
        n_blocks_w = (w + block_size - 1) // block_size
        n_blocks_d = (d + block_size - 1) // block_size
        total_blocks = n_blocks_h * n_blocks_w * n_blocks_d

        num_selected = max(1, total_blocks // 2)
        block_mask = torch.zeros(n_blocks_h, n_blocks_w, n_blocks_d, dtype=torch.bool, device=device)
        indices = torch.randperm(total_blocks, device=device)[:num_selected]
        block_mask.view(-1)[indices] = True

        mask = upsample_block_mask(block_mask, block_size)
        complementary_mask = 1.0 - mask

        masked_sc_pred = lowres_single_channel_pred * mask
        masked_mc_pred = lowres_multi_channel_pred * mask

        if disable_tta:
            first_half_pred = model(
                queries=queries,
                image_input=patches,
                simulated_lowres_sc_pred=masked_sc_pred,
                simulated_lowres_mc_pred=masked_mc_pred,
                train_mode=False
            )
        else:
            first_half_pred = internal_maybe_mirror_and_predict(
                model=model,
                queries=queries,
                image_input=patches,
                simulated_lowres_sc_pred=masked_sc_pred,
                simulated_lowres_mc_pred=masked_mc_pred,
                mirror_axes=(0, 1, 2)
            )

        masked_sc_pred_comp = lowres_single_channel_pred * complementary_mask
        masked_mc_pred_comp = lowres_multi_channel_pred * complementary_mask

        if disable_tta:
            second_half_pred = model(
                queries=queries,
                image_input=patches,
                simulated_lowres_sc_pred=masked_sc_pred_comp,
                simulated_lowres_mc_pred=masked_mc_pred_comp,
                train_mode=False
            )
        else:
            second_half_pred = internal_maybe_mirror_and_predict(
                model=model,
                queries=queries,
                image_input=patches,
                simulated_lowres_sc_pred=masked_sc_pred_comp,
                simulated_lowres_mc_pred=masked_mc_pred_comp,
                mirror_axes=(0, 1, 2)
            )

        final_prediction = first_half_pred * complementary_mask + second_half_pred * mask
        prediction_sum += final_prediction

    return prediction_sum / n_repeats


def read_npz_data(raw_image, raw_spacing, crop_size=[288, 288, 96],
                  target_spacing=[1.5, 1.5, 3.0], scaled_roi_lowres_pred_array=None,
                  class_name_list=[], stage_1_flag=False, device=torch.device("cuda", 0), verbose=True):
    """
    Read and preprocess image data for inference.
    
    This function handles spacing adjustments, image resampling, padding,
    and patch splitting for the inference pipeline.
    
    Args:
        raw_image: Input image array with shape (d, h, w)
        raw_spacing: Spacing array with shape (3,)
        crop_size: Target crop size [h, w, d]
        target_spacing: Target spacing [h, w, d]
        scaled_roi_lowres_pred_array: Optional low-res prediction for ROI-based inference
        class_name_list: List of class names (kept for compatibility, not used)
        stage_1_flag: Whether this is Stage 1 inference (kept for compatibility, not used)
        device: PyTorch device for resampling
        verbose: Whether to print detailed information (default: True)
    
    Returns:
        data_dict: Dictionary containing preprocessed patches and metadata
    """
    raw_d, raw_h, raw_w = raw_image.shape
    image = rearrange(raw_image, 'd h w -> h w d')
    spacing = raw_spacing.astype(np.float32)
    
    # Simplified spacing adjustment following the provided steps
    # Step 1: Handle very small spacing values
    for i in range(3):
        if spacing[i] <= 0.1:
            spacing[i] = 1.0
    
    # Step 2: Adjust spacing based on image dimensions
    spacing = adjust_spacing(image, spacing)
    
    # Step 3: Initialize parameters for spacing adjustment
    max_dims = [1000, 1000, 700]
    min_dims = crop_size
    thresholds = []
    current = 1.25
    while current <= 50:
        thresholds.append(current)
        current *= 1.25
    raw_target_spacing = target_spacing.copy()
    
    # Step 4: Adjust spacing based on constraints
    for i in range(3):
        # If spacing is less than 1.0 and image dimension is within max_dims, set to 1.0
        if spacing[i] < 1.0 and image.shape[i] <= max_dims[i]:
            spacing[i] = 1.0  # second stage model resolution
        
        # If physical dimension exceeds max_dims and spacing is greater than target, use target spacing
        if spacing[i] * image.shape[i] > max_dims[i] * target_spacing[i] and spacing[i] > target_spacing[i]:
            spacing[i] = target_spacing[i]
        # If physical dimension is less than min_dims threshold, adjust target_spacing
        elif spacing[i] * image.shape[i] < min_dims[i] * target_spacing[i]:
            alpha_spacing = 1
            for threshold in reversed(thresholds):
                if image.shape[i] <= (min_dims[i] / threshold):
                    alpha_spacing = threshold
                    break

            raw_target_spacing[i] = target_spacing[i]
            target_spacing[i] = max(spacing[i] * image.shape[i] / min_dims[i], spacing[i] / alpha_spacing)
            if verbose:
                print("alpha_spacing: ", alpha_spacing)
                print("spacing[i] * image.shape[i] / min_dims[i], spacing[i] / alpha_spacing: ", spacing[i] * image.shape[i] / min_dims[i], spacing[i] / alpha_spacing)
                print("raw_target_spacing[i], target_spacing[i]: ", raw_target_spacing[i], target_spacing[i])
            target_spacing[i] = min(raw_target_spacing[i], target_spacing[i])
            if verbose:
                print("image.shape[i], min_dims[i], target_spacing[i], spacing[i]: ", image.shape[i], min_dims[i], target_spacing[i], spacing[i])
    
    # Set default num_iterations (no special class handling)
    num_iterations = 1

    image = image[np.newaxis, ...].astype(np.float32)
    if verbose:
        print("image.shape: ", image.shape)
        print("spacing: ", spacing)
        print("target_spacing: ", target_spacing)
    image = respace_image(image, spacing, target_spacing, torch.device('cpu'))
    if verbose:
        print("respace image.shape: ", image.shape)
    image = torch.tensor(image)
    image, padding_info = pad_if_necessary(image, crop_size=crop_size)
    _, h, w, d = image.shape

    patches, y1y2_x1x2_z1z2_ls = split_3d(image, crop_size=crop_size)

    data_dict = {
        'spacing': spacing,
        'original_shape': (raw_h, raw_w, raw_d),
        'current_shape': (h, w, d),
        'patches': patches,
        'y1y2_x1x2_z1z2_ls': y1y2_x1x2_z1z2_ls,
        'padding_info': padding_info,
        'raw_image': raw_image,
        'num_iterations': num_iterations
    }

    if scaled_roi_lowres_pred_array is not None:
        lowres_pred = rearrange(scaled_roi_lowres_pred_array, 'd h w -> h w d')
        lowres_pred = lowres_pred[np.newaxis, ...].astype(np.float32)
        lowres_pred = respace_mask(lowres_pred, spacing, target_spacing, torch.device('cpu'))
        lowres_pred = torch.tensor(lowres_pred)
        lowres_pred, padding_info = pad_if_necessary(lowres_pred, crop_size=crop_size)
        lowres_pred_patches, _ = split_3d(lowres_pred, crop_size=crop_size)
        data_dict['lowres_pred_patches'] = lowres_pred_patches
        data_dict['padding_info'] = padding_info

    return data_dict


def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 10, dtype=np.float16):
    """
    Compute Gaussian importance map for patch weighting.
    
    This creates a Gaussian weight map centered at the patch center, used for
    weighted averaging of overlapping patch predictions.
    
    Args:
        tile_size: Size of the tile (crop_size)
        sigma_scale: Scale factor for Gaussian sigma (relative to tile size)
        value_scaling_factor: Scaling factor for the Gaussian values
        dtype: Data type for the output array
    
    Returns:
        Gaussian importance map array
    """
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])
    return gaussian_importance_map


def sc_mask_to_mc_mask(sc_mask, label_values_ls):
    """
    Convert single-channel mask to multi-channel mask.
    
    Args:
        sc_mask: Single-channel mask with shape (1, 1, h, w, d) or (h, w, d)
        label_values_ls: List of label values to create channels for
    
    Returns:
        Multi-channel mask with shape (1, n_classes, h, w, d)
    """
    sc_mask = sc_mask.squeeze(0).squeeze(0)
    assert sc_mask.ndim == 3
    h, w, d = sc_mask.shape
    n = len(label_values_ls)
    mc_mask = torch.zeros((n, h, w, d), dtype=bool).to(sc_mask.device)
    for i, label_value in enumerate(label_values_ls):
        mc_mask[i] = torch.where(sc_mask == label_value, 1, 0)
    mc_mask = mc_mask.to(torch.float32)
    mc_mask = mc_mask.unsqueeze(0)
    return mc_mask


class MedicalSegmentationPipeline:
    """
    Pipeline for medical image segmentation.
    
    This class handles model loading, data preprocessing, and inference execution
    for the Medal-S segmentation pipeline.
    """
    
    def __init__(self, config):
        """
        Initialize the segmentation pipeline.
        
        Args:
            config: Dictionary containing pipeline configuration parameters
        """
        self.config = config
        self.device = torch.device(config['device'])

    def _load_model(self):
        """
        Load vision model and text encoder from checkpoints.
        
        Returns:
            model: Loaded vision model (Maskformer)
            text_encoder: Loaded text encoder (Knowledge_Encoder)
        """
        crop_str = '_'.join(map(str, self.config['crop_size']))
        spacing_str = '_'.join(map(str, self.config['target_spacing_model']))
        
        vision_backbone_checkpoint = os.path.join(
            self.config['checkpoints_path'],
            f"nano_UNet_CVPR2025_crop_size_{crop_str}_spacing_{spacing_str}_step_{self.config['model_step']}.pth")

        model = Maskformer(
            self.config['vision_backbone'],
            self.config['input_channels'],
            self.config['crop_size'],
            self.config['patch_size'],
            False
        )
        model = model.to(self.device)
        checkpoint = torch.load(vision_backbone_checkpoint, map_location=self.device)
        new_state_dict = {
            k[7:] if k.startswith('module.') else k: v
            for k, v in checkpoint['model_state_dict'].items()
            if 'mid_mask_embed_proj' not in k
        }
        model.load_state_dict(new_state_dict)
        model.eval()

        text_encoder = Knowledge_Encoder(
            biolord_checkpoint=os.path.join(
                self.config['checkpoints_path'],
                'BioLORD-2023-C'
            )
        )
        text_encoder = text_encoder.to(self.device)
        checkpoint = torch.load(
            os.path.join(self.config['checkpoints_path'], 'text_encoder.pth'),
            map_location=self.device
        )
        new_state_dict = {
            k[7:] if k.startswith('module.') else k: v
            for k, v in checkpoint['model_state_dict'].items()
        }
        text_encoder.load_state_dict(new_state_dict, strict=False)
        text_encoder.eval()

        return model, text_encoder

    def run_inference(self, raw_image, raw_spacing, verbose=True):
        """
        Run inference on the input image.
        
        This method performs the complete inference pipeline:
        1. Load models (vision backbone and text encoder)
        2. Preprocess image data (resampling, padding, patch splitting)
        3. Encode text prompts
        4. Process patches and aggregate predictions
        5. Post-process results (remove padding, resample to original shape)
        
        Args:
            raw_image: Input image array with shape (d, h, w)
            raw_spacing: Spacing array with shape (3,)
            verbose: Whether to print detailed information (default: True)
        
        Returns:
            pred_array: Segmentation array with shape (d, h, w), dtype int16
            max_prob_array: Maximum probability array (if return_max_prob=True), or None
        """
        model, text_encoder = self._load_model()
        pred_array = None
        crop_size = self.config['crop_size']
        disable_tta = self.config['disable_tta']
        instance_label = self.config['instance_label']
        modality = self.config['modality']
        text_prompts = self.config['texts']
        label_values = self.config['label_values']
        return_max_prob = self.config['return_max_prob']
        class_name_list = self.config['class_name_list']
        stage_1_flag = self.config['stage_1_flag']
        with torch.no_grad():
            # Gaussian is kept on CPU, as accumulation will now happen on CPU
            gaussian = torch.tensor(compute_gaussian(tuple(crop_size)), dtype=torch.float32).cpu()
            
            data_dict = read_npz_data(
                raw_image=raw_image,
                raw_spacing=raw_spacing,
                crop_size=crop_size,
                target_spacing=self.config['target_spacing'],
                scaled_roi_lowres_pred_array=self.config['scaled_roi_lowres_pred_array'],
                class_name_list=class_name_list,
                stage_1_flag=stage_1_flag,
                device=self.device,
                verbose=verbose
            )

            spacing = data_dict['spacing']
            original_shape = data_dict['original_shape']
            current_shape = data_dict['current_shape']
            batched_patches = data_dict['patches']
            batched_y1y2_x1x2_z1z2 = data_dict['y1y2_x1x2_z1z2_ls']
            padding_info = data_dict['padding_info']
            raw_image = data_dict['raw_image']
            num_iterations = data_dict['num_iterations']
            batched_lowres_pred_patches = data_dict.get('lowres_pred_patches')

            modality_code = torch.tensor([{
                'ct': 0, 'mri': 1, 'us': 2, 'pet': 3, 'microscopy': 4
            }[modality]]).to(self.device)  # Keep modality_code on GPU if text_encoder needs it on GPU

            h, w, d = current_shape
            n_total_classes = len(text_prompts)
            
            # Get category batch size from config, default to 24
            category_batch_size = self.config.get('category_batch_size', 24)
            background_threshold = self.config.get('background_threshold', 0.5)
            
            # Initialize max_prob and max_class_label_value on CPU to save GPU memory
            max_prob = torch.zeros((h, w, d), dtype=torch.float32, device='cpu')
            max_class_label_value = torch.zeros((h, w, d), dtype=torch.int16, device='cpu')
            
            # Process categories in batches to avoid OOM
            category_range = range(0, n_total_classes, category_batch_size)
            pbar = tqdm(category_range, desc="Processing Categories")
            for i in pbar:
                current_category_texts = text_prompts[i:i + category_batch_size]
                current_label_values = label_values[i:i + category_batch_size]
                current_n = len(current_category_texts)
                end_idx = min(i + current_n - 1, n_total_classes - 1)
                
                # Update progress bar description with current category range
                pbar.set_description(f"Processing Categories {i}-{end_idx}")
                
                # Keep these large tensors on CPU for accumulation
                temp_prediction_batch_cpu = torch.zeros((current_n, h, w, d), dtype=torch.float32, device='cpu')
                temp_accumulation_batch_cpu = torch.zeros((current_n, h, w, d), dtype=torch.float32, device='cpu')
                
                # Encode text prompts for current batch
                with autocast(enabled=False):
                    queries = text_encoder(current_category_texts, modality_code, self.device)  # queries remain on GPU for model input
                
                # Process patches for current category batch
                for patches, lowres_pred_patches, y1y2_x1x2_z1z2_ls in tqdm(
                    zip(batched_patches, batched_lowres_pred_patches if batched_lowres_pred_patches is not None else [None]*len(batched_patches), batched_y1y2_x1x2_z1z2),
                    total=len(batched_patches),
                    desc="Processing",
                    ncols=100,
                    bar_format="{l_bar}{bar:20}{r_bar}",
                    colour="green",
                    leave=False
                ):
                    patches = patches.unsqueeze(0).to(device=self.device, dtype=torch.float32)  # patches on GPU for model input
                    y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls
                    
                    simulated_lowres_sc_pred = None
                    simulated_lowres_mc_pred = None
                    
                    if not self.config['w_lowres_pred_prompts']:
                        simulated_lowres_sc_pred = torch.zeros((1, 1, *crop_size), device=self.device, dtype=torch.float32)
                        simulated_lowres_mc_pred = torch.zeros((1, current_n, *crop_size), device=self.device, dtype=torch.float32)
                        prediction_patch = model(
                            queries=queries,
                            image_input=patches,
                            simulated_lowres_sc_pred=simulated_lowres_sc_pred,
                            simulated_lowres_mc_pred=simulated_lowres_mc_pred,
                            train_mode=False
                        ) if self.config['disable_tta'] else internal_maybe_mirror_and_predict(
                            model=model,
                            queries=queries,
                            image_input=patches,
                            simulated_lowres_sc_pred=simulated_lowres_sc_pred,
                            simulated_lowres_mc_pred=simulated_lowres_mc_pred,
                            mirror_axes=(0, 1, 2)
                        )
                    else:
                        lowres_pred_patches = lowres_pred_patches.unsqueeze(0).to(device=self.device, dtype=torch.float32)
                        simulated_lowres_sc_pred = torch.where(lowres_pred_patches > 0, torch.ones_like(lowres_pred_patches), torch.zeros_like(lowres_pred_patches))
                        simulated_lowres_mc_pred = sc_mask_to_mc_mask(lowres_pred_patches, [int(val) for val in current_label_values])
                        
                        possible_block_sizes = [8]
                        if instance_label == 1:
                            n_repeats = 1
                        else:
                            n_repeats = 1
                        prediction_patch = compute_patch_prediction(queries, patches, simulated_lowres_sc_pred, simulated_lowres_mc_pred, model, possible_block_sizes, n_repeats, disable_tta)
                    
                    if instance_label == 1:  # Instance segmentation mode
                        for _ in range(num_iterations):
                            prediction_patch_prob = torch.sigmoid(prediction_patch).detach()
                            simulated_lowres_mc_pred = torch.where(prediction_patch_prob > 0.5, 1.0, 0.0)
                            simulated_lowres_sc_pred = (simulated_lowres_mc_pred.sum(dim=1, keepdim=True) > 0).float()
                            possible_block_sizes = [4]
                            n_repeats = 1
                            prediction_patch = compute_patch_prediction(queries, patches, simulated_lowres_sc_pred, simulated_lowres_mc_pred, model, possible_block_sizes, n_repeats, disable_tta)
                    
                    prediction_patch_prob_gpu = torch.sigmoid(prediction_patch).detach()
                    current_gaussian_slice = gaussian[:y2-y1, :x2-x1, :z2-z1]  # Already on CPU
                    
                    # Perform accumulation on CPU. Move prediction_patch_prob_gpu to CPU here.
                    temp_prediction_batch_cpu[:, y1:y2, x1:x2, z1:z2] += (prediction_patch_prob_gpu[0, :, :y2-y1, :x2-x1, :z2-z1].cpu() * current_gaussian_slice)
                    temp_accumulation_batch_cpu[:, y1:y2, x1:x2, z1:z2] += current_gaussian_slice
                    
                    # Explicitly delete GPU tensors to free up memory immediately
                    del prediction_patch, prediction_patch_prob_gpu, patches
                    if simulated_lowres_sc_pred is not None:
                        del simulated_lowres_sc_pred
                    if simulated_lowres_mc_pred is not None:
                        del simulated_lowres_mc_pred
                    torch.cuda.empty_cache()  # Clear any cached GPU memory after each patch processing
                    gc.collect()  # Python garbage collection
                
                # Normalize predictions by accumulation
                batch_accumulation_cpu = temp_accumulation_batch_cpu
                batch_accumulation_cpu[batch_accumulation_cpu == 0] = 1e-8
                batch_prediction_prob_cpu = temp_prediction_batch_cpu / batch_accumulation_cpu
                
                # Update max_prob and max_class_label_value on CPU
                for j in range(current_n):
                    class_prob_cpu = batch_prediction_prob_cpu[j, ...]  # Already on CPU
                    class_label_value_cpu_scalar = torch.tensor(int(current_label_values[j]), dtype=torch.int16, device='cpu')  # Already on CPU
                    
                    update_mask_cpu = class_prob_cpu > max_prob
                    max_prob[update_mask_cpu] = class_prob_cpu[update_mask_cpu]
                    max_class_label_value[update_mask_cpu] = class_label_value_cpu_scalar
                
                # Clean up batch tensors
                del temp_prediction_batch_cpu, temp_accumulation_batch_cpu, batch_accumulation_cpu, batch_prediction_prob_cpu, queries
                # Previous patch-level deletions handle GPU memory
            
            # Final operations on CPU
            background_indices = max_prob < background_threshold
            max_class_label_value[background_indices] = 0
            results = max_class_label_value.numpy()  # Already on CPU, just convert to numpy
                       
            results = remove_padding(results, padding_info)
            current_h, current_w, current_d = results.shape
            if results.shape != original_shape:
                results = resample_torch_simple(
                    results[np.newaxis, ...],
                    new_shape=original_shape,
                    is_seg=True,
                    num_threads=4,
                    device=torch.device('cpu'),
                    memefficient_seg_resampling=False).squeeze(0)
                
                if verbose:
                    print(f"Resized segmentation from {current_h, current_w, current_d} to {original_shape}")

            pred_array = rearrange(results, 'h w d -> d h w').astype(np.int16)

            if return_max_prob and instance_label == 0:
                # max_prob is already on CPU, just convert to numpy for post-processing
                max_prob_numpy = max_prob.numpy()
                max_prob_numpy = remove_padding(max_prob_numpy, padding_info)
                current_h, current_w, current_d = max_prob_numpy.shape
                if max_prob_numpy.shape != original_shape:
                    max_prob_numpy = resample_torch_simple(
                        max_prob_numpy[np.newaxis, ...],
                        new_shape=original_shape,
                        is_seg=False,
                        num_threads=4,
                        device=torch.device('cpu'),
                        memefficient_seg_resampling=False).squeeze(0)

                    if verbose:
                        print(f"Resized max probability from {current_h, current_w, current_d} to {original_shape}")
                max_prob = rearrange(max_prob_numpy, 'h w d -> d h w').astype(np.float32)
        
        if return_max_prob and instance_label == 0:
            return pred_array, max_prob
        else:
            return pred_array, None


def run_segmentation(
    raw_image,
    raw_spacing,
    crop_size=[192, 192, 96],
    target_spacing=[1.5, 1.5, 3.0],
    target_spacing_model=[1.5, 1.5, 3.0],
    w_lowres_pred_prompts=False,
    scaled_roi_lowres_pred_array=None,
    disable_tta=True,
    model_step=100000,
    vision_backbone="UNET",
    input_channels=2,
    patch_size=[32, 32, 32],
    modality='CT',
    instance_label=0,
    texts=[],
    label_values=[],
    return_max_prob=False,
    class_name_list=[],
    stage_1_flag=False,
    device="cuda:0",
    checkpoints_path="./checkpoints",
    category_batch_size=24,
    background_threshold=0.5,
    verbose=True,
):
    """
    Main segmentation function.
    
    This function orchestrates the entire segmentation pipeline including
    model loading, data preprocessing, patch-based inference, and result aggregation.
    
    Args:
        raw_image: Input image array with shape (d, h, w), dtype uint8, values in [0, 255]
        raw_spacing: Spacing array with shape (3,)
        crop_size: Crop size for patch processing [h, w, d]
        target_spacing: Target spacing for resampling [h, w, d]
        target_spacing_model: Target spacing for model (should match target_spacing)
        w_lowres_pred_prompts: Whether to use low-res predictions as spatial prompts
        scaled_roi_lowres_pred_array: Low-res prediction array for spatial prompts
        disable_tta: Disable test-time augmentation
        model_step: Model checkpoint step number
        vision_backbone: Vision backbone architecture name
        input_channels: Number of input channels
        patch_size: Patch size for the model
        modality: Imaging modality ('CT', 'MRI', 'US', 'PET', 'microscopy')
        instance_label: 0 for semantic segmentation, 1 for instance segmentation
        texts: List of text prompts (one per class)
        label_values: List of label values (one per class)
        return_max_prob: Whether to return maximum probability map
        class_name_list: List of class names for class-specific adjustments
        stage_1_flag: Whether this is Stage 1 inference
        device: Device string (e.g., 'cuda:0' or 'cpu')
        checkpoints_path: Path to model checkpoints directory
        category_batch_size: Number of categories to process in each batch (default: 24)
            Adjust based on GPU memory. Larger 3D images require smaller batch sizes.
            Accumulation operations are performed on CPU for more stable memory usage.
        background_threshold: Probability threshold for background (default: 0.5)
            Voxels with max probability below this threshold will be labeled as background.
        verbose: Whether to print detailed information (default: True)
    
    Returns:
        pred_array: Segmentation array with shape (d, h, w), dtype int16
        max_prob_array: Maximum probability array (if return_max_prob=True), or None
    """
    w_lowres_pred_prompts = scaled_roi_lowres_pred_array is not None
    config = {
        'device': device,
        'modality': modality,
        'instance_label': instance_label,
        'texts': texts,
        'label_values': label_values,
        'vision_backbone': vision_backbone,
        'crop_size': crop_size,
        'patch_size': patch_size,
        'target_spacing': target_spacing,
        'target_spacing_model': target_spacing_model,
        'model_step': model_step,
        'input_channels': input_channels,
        'w_lowres_pred_prompts': w_lowres_pred_prompts,
        'scaled_roi_lowres_pred_array': scaled_roi_lowres_pred_array,
        'disable_tta': disable_tta,
        'checkpoints_path': checkpoints_path,
        'return_max_prob': return_max_prob,
        'class_name_list': class_name_list,
        'stage_1_flag': stage_1_flag,
        'category_batch_size': category_batch_size,
        'background_threshold': background_threshold,
    }
    
    pipeline = MedicalSegmentationPipeline(config)
    return pipeline.run_inference(raw_image, raw_spacing, verbose=verbose)


# ============================================================================
# Main Inference Functions
# ============================================================================
# These functions provide the high-level interface for running inference
# on raw NIfTI images with proper preprocessing and post-processing.
# ============================================================================


def normalize_image_ct(image_data, window_level=40, window_width=400, window_type='soft_tissue'):
    """
    Normalize CT image using window/level technique.
    
    Args:
        image_data: Input CT image array
        window_level: Window level (center of the window). If None, will use default based on window_type
        window_width: Window width (range of the window). If None, will use default based on window_type
        window_type: Type of window ('soft_tissue', 'bone', 'lung'). Used if window_level/window_width are None
    
    Returns:
        Normalized image array with dtype uint8, values in [0, 255]
    """
    # Default window settings for different window types
    default_windows = {
        'soft_tissue': {'window_level': 40, 'window_width': 400},
        'bone': {'window_level': 500, 'window_width': 1500},
        'lung': {'window_level': -600, 'window_width': 1500}
    }
    
    # Use defaults if not provided
    if window_level is None or window_width is None:
        if window_type in default_windows:
            window_level = default_windows[window_type]['window_level']
            window_width = default_windows[window_type]['window_width']
        else:
            # Fallback to soft_tissue defaults
            window_level = default_windows['soft_tissue']['window_level']
            window_width = default_windows['soft_tissue']['window_width']
    
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / (np.max(image_data_pre) - np.min(image_data_pre) + 1e-8)
        * 255.0
    )
    return image_data_pre.astype(np.uint8)


def normalize_image_other(image_data, percentile_lower=None, percentile_upper=None, preserve_zero=None, normalization_settings=None):
    """
    Normalize non-CT images using percentile-based normalization.
    
    This method clips values to specified percentiles, then
    normalizes to [0, 255] range while optionally preserving zero values.
    
    Args:
        image_data: Input image array
        percentile_lower: Lower percentile for clipping. If None, will use default or value from normalization_settings
        percentile_upper: Upper percentile for clipping. If None, will use default or value from normalization_settings
        preserve_zero: Whether to preserve zero values. If None, will use default or value from normalization_settings
        normalization_settings: Dictionary containing normalization settings from config.
            Format: {'percentile_lower': 0.5, 'percentile_upper': 99.5, 'preserve_zero': True}
    
    Returns:
        Normalized image array with dtype uint8, values in [0, 255]
    """
    # Default normalization settings
    default_percentile_lower = 0.5
    default_percentile_upper = 99.5
    default_preserve_zero = True
    
    # Use settings from config if provided
    if normalization_settings is not None:
        if percentile_lower is None:
            percentile_lower = normalization_settings.get('percentile_lower', default_percentile_lower)
        if percentile_upper is None:
            percentile_upper = normalization_settings.get('percentile_upper', default_percentile_upper)
        if preserve_zero is None:
            preserve_zero = normalization_settings.get('preserve_zero', default_preserve_zero)
    else:
        # Use defaults if not provided
        if percentile_lower is None:
            percentile_lower = default_percentile_lower
        if percentile_upper is None:
            percentile_upper = default_percentile_upper
        if preserve_zero is None:
            preserve_zero = default_preserve_zero
    
    # Calculate percentiles from non-zero values
    non_zero_data = image_data[image_data > 0]
    if len(non_zero_data) > 0:
        lower_bound, upper_bound = np.percentile(
            non_zero_data, [percentile_lower, percentile_upper]
        )
    else:
        # If all values are zero, use min/max
        lower_bound = np.min(image_data)
        upper_bound = np.max(image_data)
    
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / (np.max(image_data_pre) - np.min(image_data_pre) + 1e-8)
        * 255.0
    )
    
    if preserve_zero:
        image_data_pre[image_data == 0] = 0
    
    return image_data_pre.astype(np.uint8)


def load_nifti_image(image_path):
    """
    Load NIfTI image and extract data, spacing, and metadata.
    
    Args:
        image_path: Path to NIfTI image file
    
    Returns:
        image_data: Image array with shape (d, h, w)
        spacing_xyz: Spacing tuple (x, y, z) from SimpleITK
        metadata: Dictionary containing origin, direction, and spacing_xyz
    """
    img_sitk = sitk.ReadImage(image_path)
    image_data = sitk.GetArrayFromImage(img_sitk)  # Shape: (d, h, w)
    spacing_xyz = img_sitk.GetSpacing()  # (x, y, z)
    
    # Save metadata for output
    metadata = {
        'origin': img_sitk.GetOrigin(),
        'direction': img_sitk.GetDirection(),
        'spacing_xyz': spacing_xyz
    }
    
    return image_data, spacing_xyz, metadata


def convert_spacing(spacing_xyz, image_shape):
    """
    Convert spacing from SimpleITK format (x, y, z) to format expected by run_segmentation.
    
    Following the conversion logic from inference_raw_nifti_2.py:
    1. SimpleITK returns (x, y, z)
    2. Image from SimpleITK is (d, h, w) where d=z, h=y, w=x
    3. Convert to (d, h, w) spacing: (z, x, y) = (d, h, w)
    4. Then convert to format expected by run_segmentation: (h, w, d)
    
    Args:
        spacing_xyz: Spacing tuple from SimpleITK (x, y, z)
        image_shape: Image shape (d, h, w)
    
    Returns:
        img_spacing: Spacing array in format expected by run_segmentation
    """
    img_spacing = np.array(spacing_xyz, dtype=np.float32)
    
    # Step 1: Convert from (x, y, z) to (d, h, w) spacing
    # SimpleITK: (x, y, z) -> Image: (d, h, w) where d=z, h=y, w=x
    # So spacing (x, y, z) -> (z, x, y) = (d, h, w)
    img_spacing_transposed = img_spacing[[2, 0, 1]]  # (z, x, y) = (d, h, w)
    
    # Step 2: Handle very small spacing values
    for i in range(3):
        if img_spacing_transposed[i] < 0.1:
            img_spacing_transposed[i] = 1.0
    
    # Step 3: Optional: Adjust spacing based on image dimensions
    # Note: adjust_spacing expects image in (h, w, d) format, so we need to rearrange
    # For now, we'll skip this adjustment or use a dummy array
    try:
        img_spacing_transposed = adjust_spacing(
            np.zeros(image_shape),  # Dummy array for shape reference
            img_spacing_transposed
        ).astype(np.float32)
    except Exception:
        # If adjust_spacing fails, use spacing as-is
        pass
    
    # Step 4: Convert to format expected by run_segmentation
    # This converts (d, h, w) to (h, w, d)
    img_spacing = img_spacing_transposed[[1, 2, 0]]
    
    return img_spacing


def run_inference_single_window(
    image_data,
    spacing_xyz,
    metadata,
    modality='CT',
    texts=None,
    label_values=None,
    inference_mode='stage2_only',
    device="cuda:0",
    checkpoints_path="./checkpoints",
    window_settings=None,
    window_type='soft_tissue',
    normalization_settings=None,
    verbose=True
):
    """
    Run inference for a single window type.
    
    This is an internal function used by run_inference to handle single window type inference.
    
    Args:
        image_data: Raw image data array (d, h, w)
        spacing_xyz: Spacing tuple (x, y, z)
        metadata: Image metadata dictionary
        modality: Imaging modality ('CT', 'MRI', 'US', 'PET', 'microscopy')
        texts: List of text prompts (one per class)
        label_values: List of label values (one per class)
        inference_mode: Inference mode ('stage2_only' or 'stage1+stage2')
        device: Device to use ('cuda:0' or 'cpu')
        checkpoints_path: Path to model checkpoints
        window_settings: Dictionary containing window settings for different window types (CT only)
        window_type: Type of window to use ('soft_tissue', 'bone', 'lung')
        normalization_settings: Dictionary containing normalization settings for non-CT modalities
        verbose: Whether to print detailed information (default: True)
    
    Returns:
        pred_array: Segmentation array (d, h, w)
    """
    if texts is None:
        texts = []
    if label_values is None:
        label_values = []
    
    if len(texts) != len(label_values):
        raise ValueError("Number of text prompts must match number of label values")
    
    # Normalize image
    if verbose:
        print(f"Normalizing image for {window_type} window (modality: {modality})")
    if modality.upper() == 'CT':
        # Get window settings from config if available
        window_level = None
        window_width = None
        if window_settings is not None and window_type in window_settings:
            window_level = window_settings[window_type].get('window_level')
            window_width = window_settings[window_type].get('window_width')
            if verbose:
                print(f"Using {window_type} window: level={window_level}, width={window_width}")
        
        img_array = normalize_image_ct(image_data, window_level=window_level, 
                                       window_width=window_width, window_type=window_type)
    else:
        # Get normalization settings from config if available
        if normalization_settings is not None:
            if verbose:
                print(f"Using normalization settings from config: {normalization_settings}")
            img_array = normalize_image_other(image_data, normalization_settings=normalization_settings)
        else:
            # Use default normalization
            if verbose:
                print("Using default normalization settings")
            img_array = normalize_image_other(image_data)
    
    if verbose:
        print(f"Normalized image range: [{img_array.min()}, {img_array.max()}]")
    
    # Convert spacing
    img_spacing = convert_spacing(spacing_xyz, img_array.shape)
    if verbose:
        print(f"Converted spacing: {img_spacing}")
    
    # Run inference
    if inference_mode == 'stage1+stage2':
        if verbose:
            print(f"Running two-stage inference with {window_type} window...")
        # Stage 1: Low-resolution
        if verbose:
            print("Stage 1: Low-resolution segmentation...")
        stage_1_pred, _ = run_segmentation(
            raw_image=img_array,
            raw_spacing=img_spacing,
            crop_size=[224, 224, 128],
            target_spacing=[1.5, 1.5, 3.0],
            target_spacing_model=[1.5, 1.5, 3.0],
            w_lowres_pred_prompts=False,
            scaled_roi_lowres_pred_array=None,
            disable_tta=True,
            model_step=358600,
            modality=modality.lower(),
            instance_label=0,
            texts=texts,
            label_values=label_values,
            return_max_prob=False,
            class_name_list=[],
            stage_1_flag=True,
            device=device,
            checkpoints_path=checkpoints_path,
            verbose=verbose
        )
        
        # Check if Stage 1 found anything
        if stage_1_pred.sum() == 0:
            if verbose:
                print("Warning: Stage 1 found no predictions. Using Stage 1 result as final output.")
            final_pred = stage_1_pred
        else:
            if verbose:
                print("Stage 1 completed. Extracting ROI for Stage 2...")
            
            # Remove small objects from Stage 1 prediction
            min_size = 10
            lowres_pred_binary = (stage_1_pred > 0).astype(np.int16)
            lowres_pred_binary = remove_small_objects_binary(lowres_pred_binary, min_size=min_size).astype(np.int16)
            stage_1_pred_cleaned = stage_1_pred * lowres_pred_binary
            
            # Extract ROI from Stage 1 prediction
            # Find bounding box of non-zero regions
            non_zero_indices = np.argwhere(stage_1_pred_cleaned > 0)
            if len(non_zero_indices) == 0:
                if verbose:
                    print("Warning: No non-zero regions after cleaning. Using Stage 1 result.")
                final_pred = stage_1_pred_cleaned
            else:
                z_min, y_min, x_min = non_zero_indices.min(axis=0)
                z_max, y_max, x_max = non_zero_indices.max(axis=0)
                
                # Calculate ROI center and range with scaling factor
                m = 1.1  # Scaling factor for ROI expansion
                z_center = (z_min + z_max) / 2
                y_center = (y_min + y_max) / 2
                x_center = (x_min + x_max) / 2
                
                z_range = (z_max - z_min + 1) * m / 2
                y_range = (y_max - y_min + 1) * m / 2
                x_range = (x_max - x_min + 1) * m / 2
                
                # Calculate minimum ranges based on Stage 2 crop size and spacing
                stage_2_crop_size = [192, 192, 192]
                stage_2_target_spacing = [1.0, 1.0, 1.0]
                
                img_spacing_for_roi = img_spacing.copy()
                
                min_z_range = (stage_2_crop_size[2] / 2) * stage_2_target_spacing[2] / img_spacing_for_roi[2] if img_spacing_for_roi[2] > 0 else z_range
                min_y_range = (stage_2_crop_size[0] / 2) * stage_2_target_spacing[0] / img_spacing_for_roi[0] if img_spacing_for_roi[0] > 0 else y_range
                min_x_range = (stage_2_crop_size[1] / 2) * stage_2_target_spacing[1] / img_spacing_for_roi[1] if img_spacing_for_roi[1] > 0 else x_range
                
                z_range = max(min_z_range - 1, z_range)
                y_range = max(min_y_range - 1, y_range)
                x_range = max(min_x_range - 1, x_range)
                
                z_min_new = max(0, int(z_center - z_range))
                z_max_new = min(stage_1_pred_cleaned.shape[0] - 1, int(z_center + z_range))
                y_min_new = max(0, int(y_center - y_range))
                y_max_new = min(stage_1_pred_cleaned.shape[1] - 1, int(y_center + y_range))
                x_min_new = max(0, int(x_center - x_range))
                x_max_new = min(stage_1_pred_cleaned.shape[2] - 1, int(x_center + x_range))
                
                if verbose:
                    print(f"ROI bounds: z=[{z_min_new}:{z_max_new}], y=[{y_min_new}:{y_max_new}], x=[{x_min_new}:{x_max_new}]")
                
                roi_array = img_array[z_min_new:z_max_new+1, y_min_new:y_max_new+1, x_min_new:x_max_new+1]
                roi_lowres_pred = stage_1_pred_cleaned[z_min_new:z_max_new+1, y_min_new:y_max_new+1, x_min_new:x_max_new+1]
                
                if verbose:
                    print(f"ROI image shape: {roi_array.shape}")
                    print(f"ROI prediction shape: {roi_lowres_pred.shape}")
                
                # Stage 2: High-resolution segmentation on ROI
                if verbose:
                    print("Stage 2: High-resolution segmentation on ROI...")
                roi_pred, _ = run_segmentation(
                    raw_image=roi_array,
                    raw_spacing=img_spacing,
                    crop_size=[192, 192, 192],
                    target_spacing=[1.0, 1.0, 1.0],
                    target_spacing_model=[1.0, 1.0, 1.0],
                    w_lowres_pred_prompts=True,
                    scaled_roi_lowres_pred_array=roi_lowres_pred,
                    disable_tta=True,
                    model_step=341300,
                    modality=modality.lower(),
                    instance_label=0,
                    texts=texts,
                    label_values=label_values,
                    return_max_prob=False,
                    class_name_list=[],
                    stage_1_flag=False,
                    device=device,
                    checkpoints_path=checkpoints_path,
                    verbose=verbose
                )
                
                # Integrate ROI prediction back into full volume
                if verbose:
                    print("Integrating Stage 2 results back into full volume...")
                final_pred = np.zeros_like(stage_1_pred_cleaned, dtype=np.int16)
                final_pred[z_min_new:z_max_new+1, y_min_new:y_max_new+1, x_min_new:x_max_new+1] = roi_pred
                if verbose:
                    print("Stage1+Stage2 inference completed.")
    elif inference_mode == 'stage2_only':
        if verbose:
            print(f"Running Stage 2 inference with {window_type} window...")
        final_pred, _ = run_segmentation(
            raw_image=img_array,
            raw_spacing=img_spacing,
            crop_size=[192, 192, 192],
            target_spacing=[1.0, 1.0, 1.0],
            target_spacing_model=[1.0, 1.0, 1.0],
            w_lowres_pred_prompts=False,
            scaled_roi_lowres_pred_array=None,
            disable_tta=True,
            model_step=341300,
            modality=modality.lower(),
            instance_label=0,
            texts=texts,
            label_values=label_values,
            return_max_prob=False,
            class_name_list=[],
            stage_1_flag=False,
            device=device,
            checkpoints_path=checkpoints_path,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown inference mode: {inference_mode}. Must be 'stage2_only' or 'stage1+stage2'")
    
    return final_pred


def run_inference(
    image_path,
    output_path,
    modality='CT',
    texts=None,
    label_values=None,
    inference_mode='stage2_only',
    device="cuda:0",
    checkpoints_path="./checkpoints",
    window_settings=None,
    window_type='soft_tissue',
    normalization_settings=None,
    window_type_mapping=None,
    verbose=True
):
    """
    Run Medal-S inference on a raw NIfTI image.
    
    Supports multi-window inference for CT images: if multiple window types are specified
    (e.g., soft_tissue, bone, lung), each window type will be processed separately with
    its corresponding window settings, and results will be merged.
    
    Args:
        image_path: Path to input NIfTI image
        output_path: Path to save output segmentation (will be modified with mode suffix)
        modality: Imaging modality ('CT', 'MRI', 'US', 'PET', 'microscopy')
        texts: List of text prompts (one per class)
        label_values: List of label values (one per class)
        inference_mode: Inference mode ('stage2_only' or 'stage1+stage2')
        device: Device to use ('cuda:0' or 'cpu')
        checkpoints_path: Path to model checkpoints
        window_settings: Dictionary containing window settings for different window types (CT only).
            Format: {'soft_tissue': {'window_level': 40, 'window_width': 400}, ...}
        window_type: Type of window to use ('soft_tissue', 'bone', 'lung'). Default: 'soft_tissue' (CT only)
            Ignored if window_type_mapping indicates multiple window types
        normalization_settings: Dictionary containing normalization settings for non-CT modalities.
            Format: {'percentile_lower': 0.5, 'percentile_upper': 99.5, 'preserve_zero': True}
        window_type_mapping: Dictionary mapping each text to its window type.
            Format: {'text1': 'soft_tissue', 'text2': 'bone', ...}
            If provided and contains multiple window types, will perform separate inference for each
        verbose: Whether to print detailed information (default: True)
    
    Returns:
        pred_array: Segmentation array (d, h, w)
        inference_time: Total inference time in seconds
    """
    if texts is None:
        texts = []
    if label_values is None:
        label_values = []
    
    if len(texts) != len(label_values):
        raise ValueError("Number of text prompts must match number of label values")
    
    # Add mode suffix to output filename
    if inference_mode == 'stage1+stage2':
        suffix = '_stage1+stage2'
    elif inference_mode == 'stage2_only':
        suffix = '_stage2_only'
    else:
        suffix = f'_{inference_mode}'
    
    # Modify output path to include suffix
    base_path, ext = os.path.splitext(output_path)
    if ext == '.gz':  # Handle .nii.gz
        base_path, nii_ext = os.path.splitext(base_path)
        output_path = f"{base_path}{suffix}{nii_ext}{ext}"
    else:
        output_path = f"{base_path}{suffix}{ext}"
    
    if verbose:
        print(f"Output will be saved to: {output_path}")
    
    # Start timing
    start_time = time.time()
    
    # Load image
    if verbose:
        print(f"Loading image: {image_path}")
    image_data, spacing_xyz, metadata = load_nifti_image(image_path)
    if verbose:
        print(f"Image shape: {image_data.shape}")
        print(f"Original spacing (x, y, z): {spacing_xyz}")
    
    # Determine inference strategy based on modality and window types
    if modality.upper() == 'CT':
        # CT modality: check for multiple window types
        if window_type_mapping is not None:
            window_types = list(set(window_type_mapping.values()))
            if len(window_types) > 1:
                # Multiple window types: perform separate inference for each window type
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"CT with {len(window_types)} window types detected: {window_types}")
                    print("Performing separate inference for each window type...")
                    print(f"{'='*60}\n")
                
                all_predictions = []
                
                for wt in window_types:
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"Processing {wt} window type...")
                        print(f"{'='*60}\n")
                    
                    # Filter texts and label_values for this window type
                    wt_texts = [text for text in texts if window_type_mapping.get(text) == wt]
                    wt_indices = [i for i, text in enumerate(texts) if window_type_mapping.get(text) == wt]
                    wt_label_values = [label_values[i] for i in wt_indices]
                    
                    if len(wt_texts) == 0:
                        if verbose:
                            print(f"No classes for {wt} window type, skipping...")
                        continue
                    
                    if verbose:
                        print(f"Classes for {wt} window: {len(wt_texts)}")
                        print(f"  Texts: {wt_texts}")
                        print(f"  Labels: {wt_label_values}")
                    
                    # Run inference for this window type with its specific window settings
                    wt_pred = run_inference_single_window(
                        image_data=image_data,
                        spacing_xyz=spacing_xyz,
                        metadata=metadata,
                        modality=modality,
                        texts=wt_texts,
                        label_values=wt_label_values,
                        inference_mode=inference_mode,
                        device=device,
                        checkpoints_path=checkpoints_path,
                        window_settings=window_settings,
                        window_type=wt,  # Use the specific window type
                        normalization_settings=normalization_settings,
                        verbose=verbose
                    )
                    
                    all_predictions.append((wt_pred, wt_label_values))
                
                # Merge predictions: use maximum label value when overlapping
                if verbose:
                    print(f"\n{'='*60}")
                    print("Merging predictions from all window types...")
                    print(f"{'='*60}\n")
                
                final_pred = np.zeros_like(all_predictions[0][0], dtype=np.int16)
                for wt_pred, wt_labels in all_predictions:
                    # For each label in this window type's prediction
                    for label_val in wt_labels:
                        label_int = int(label_val)
                        mask = (wt_pred == label_int)
                        # Only update if current prediction is background (0) or smaller label
                        final_pred[mask] = np.maximum(final_pred[mask], label_int)
                
                if verbose:
                    print("Merging completed.")
            else:
                # Single window type: use the specific window type
                if len(window_types) == 1:
                    window_type = window_types[0]
                    if verbose:
                        print(f"CT with single window type: {window_type}")
                
                final_pred = run_inference_single_window(
                    image_data=image_data,
                    spacing_xyz=spacing_xyz,
                    metadata=metadata,
                    modality=modality,
                    texts=texts,
                    label_values=label_values,
                    inference_mode=inference_mode,
                    device=device,
                    checkpoints_path=checkpoints_path,
                    window_settings=window_settings,
                    window_type=window_type,  # Use the determined window type
                    normalization_settings=normalization_settings,
                    verbose=verbose
                )
        else:
            # No window_type_mapping: use default window_type
            if verbose:
                print(f"CT without window_type_mapping, using window type: {window_type}")
            final_pred = run_inference_single_window(
                image_data=image_data,
                spacing_xyz=spacing_xyz,
                metadata=metadata,
                modality=modality,
                texts=texts,
                label_values=label_values,
                inference_mode=inference_mode,
                device=device,
                checkpoints_path=checkpoints_path,
                window_settings=window_settings,
                window_type=window_type,
                normalization_settings=normalization_settings,
                verbose=verbose
            )
    else:
        # Non-CT modality: use normalization_settings (other normalization)
        if verbose:
            print(f"Non-CT modality ({modality}): using normalization_settings")
        final_pred = run_inference_single_window(
            image_data=image_data,
            spacing_xyz=spacing_xyz,
            metadata=metadata,
            modality=modality,
            texts=texts,
            label_values=label_values,
            inference_mode=inference_mode,
            device=device,
            checkpoints_path=checkpoints_path,
            window_settings=window_settings,  # Not used for non-CT
            window_type=window_type,  # Not used for non-CT
            normalization_settings=normalization_settings,  # Used for non-CT
            verbose=verbose
        )
    
    # End timing
    end_time = time.time()
    inference_time = end_time - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Inference Mode: {inference_mode}")
        print(f"Total Inference Time: {inference_time:.2f} seconds ({inference_time/60:.2f} minutes)")
        print(f"{'='*60}\n")
    
    # Save result
    if verbose:
        print(f"Saving segmentation to: {output_path}")
    seg_sitk = sitk.GetImageFromArray(final_pred.astype(np.int16))
    seg_sitk.SetSpacing(metadata['spacing_xyz'])
    seg_sitk.SetOrigin(metadata['origin'])
    seg_sitk.SetDirection(metadata['direction'])
    sitk.WriteImage(seg_sitk, output_path)
    if verbose:
        print(f"Successfully saved segmentation to: {output_path}")
    
    return final_pred, inference_time


def load_config_from_json(config_path):
    """
    Load configuration from JSON file.
    
    Supports two formats:
    1. Legacy format: single 'texts' array
    2. New format: separate arrays for 'texts_soft_tissue', 'texts_bone', 'texts_lung'
    
    If 'labels' field is missing or empty, automatically generates consecutive
    integer labels starting from 1 (i.e., [1, 2, 3, ..., n] where n is the
    number of texts).
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        config: Dictionary containing configuration parameters with processed labels
    
    Example:
        # Legacy format:
        {"texts": ["Aorta", "Liver"], "labels": [1, 2]}
        
        # New format with window types:
        {
            "texts_soft_tissue": ["Aorta", "Liver"],
            "texts_bone": ["Vertebrae C1"],
            "texts_lung": ["Left lung"],
            "window_settings": {
                "soft_tissue": {"window_level": 40, "window_width": 400},
                "bone": {"window_level": 400, "window_width": 1500},
                "lung": {"window_level": -600, "window_width": 1500}
            }
        }
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Check if using new format (separate window types)
    has_window_types = any(key in config for key in ['texts_soft_tissue', 'texts_bone', 'texts_lung'])
    
    if has_window_types:
        # New format: combine all texts from different window types
        texts_soft_tissue = config.get('texts_soft_tissue', [])
        texts_bone = config.get('texts_bone', [])
        texts_lung = config.get('texts_lung', [])
        
        # Combine all texts in order: soft_tissue, bone, lung
        texts = texts_soft_tissue + texts_bone + texts_lung
        
        # Store window type mapping for each text
        window_type_mapping = {}
        for text in texts_soft_tissue:
            window_type_mapping[text] = 'soft_tissue'
        for text in texts_bone:
            window_type_mapping[text] = 'bone'
        for text in texts_lung:
            window_type_mapping[text] = 'lung'
        
        config['texts'] = texts
        config['window_type_mapping'] = window_type_mapping
    else:
        # Legacy format: single texts array
        texts = config.get('texts', [])
        # Default all texts to soft_tissue window type for backward compatibility
        window_type_mapping = {text: 'soft_tissue' for text in texts}
        config['window_type_mapping'] = window_type_mapping
    
    # Process labels: auto-generate if missing or empty
    texts = config.get('texts', [])
    labels = config.get('labels', None)
    
    if labels is None or len(labels) == 0:
        # Auto-generate consecutive labels starting from 1
        labels = list(range(1, len(texts) + 1))
        print(f"  Auto-generated consecutive labels: {labels}")
    else:
        # Convert labels to integers (handle both string and integer inputs)
        labels = [int(label) for label in labels]
    
    # Validate that number of labels matches number of texts
    if len(labels) != len(texts):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of texts ({len(texts)}). "
            f"Texts: {len(texts)}, Labels: {len(labels)}"
        )
    
    config['labels'] = labels
    return config


def main():
    """
    Main entry point for the inference script.
    
    Parses command-line arguments and runs inference with the specified
    configuration.
    """
    parser = argparse.ArgumentParser(
        description="Medal-S inference for raw NIfTI images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using JSON configuration file:
    python inference_medals.py --input image.nii.gz --output result.nii.gz \\
        --config config.json --mode stage2_only
    
    # Using command-line arguments:
    python inference_medals.py --input image.nii.gz --output result.nii.gz \\
        --modality CT --texts "Aorta in CT" --labels 1 --mode stage1+stage2
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input NIfTI image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save output segmentation (suffix will be added automatically based on inference mode)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to JSON configuration file (if provided, will override --texts, --labels, --modality)"
    )
    parser.add_argument(
        "--modality", "-m",
        type=str,
        default="CT",
        choices=['CT', 'MRI', 'US', 'PET', 'microscopy'],
        help="Imaging modality (default: CT, ignored if --config is provided)"
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs='+',
        default=None,
        help="Text prompts (one per class, ignored if --config is provided)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        default=None,
        help="Label values (one per class, must match texts, ignored if --config is provided)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="stage2_only",
        choices=['stage2_only', 'stage1+stage2'],
        help="Inference mode: 'stage2_only' (default) or 'stage1+stage2'"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints",
        help="Path to model checkpoints (default: ./checkpoints)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        default=False,
        help="Print detailed information during inference (default: False)"
    )
    
    args = parser.parse_args()
    verbose = args.verbose
    
    # Load configuration from JSON file if provided
    window_settings = None
    window_type = 'soft_tissue'
    normalization_settings = None
    window_type_mapping = None
    
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        config = load_config_from_json(args.config)
        texts = config.get('texts', [])
        labels = config.get('labels', [])
        modality = config.get('modality', 'CT')
        window_settings = config.get('window_settings')
        normalization_settings = config.get('normalization_settings')
        window_type_mapping = config.get('window_type_mapping')
        
        # Determine default window type based on texts (for CT only, used as fallback)
        if modality.upper() == 'CT':
            if window_type_mapping:
                window_types = list(set(window_type_mapping.values()))
                if len(window_types) == 1:
                    window_type = window_types[0]
                else:
                    # Default to soft_tissue if mixed types (will be handled by multi-window inference)
                    window_type = 'soft_tissue'
        
        # Convert labels to strings for compatibility with run_segmentation
        # (run_segmentation expects string labels)
        label_values = [str(label) for label in labels]
        
        if verbose:
            print(f"Loaded configuration from: {args.config}")
            print(f"  Modality: {modality}")
            print(f"  Number of classes: {len(texts)}")
            print(f"  Labels: {labels}")
            if modality.upper() == 'CT' and window_settings:
                print(f"  Window settings available for: {list(window_settings.keys())}")
                if window_type_mapping:
                    window_types = list(set(window_type_mapping.values()))
                    if len(window_types) > 1:
                        print(f"  Multiple window types detected: {window_types}")
                        print(f"  Will perform separate inference for each window type")
                    else:
                        print(f"  Using window type: {window_type}")
                else:
                    print(f"  Using window type: {window_type}")
            elif normalization_settings:
                print(f"  Normalization settings: {normalization_settings}")
    else:
        # Use command line arguments
        if args.texts is None or args.labels is None:
            raise ValueError("Either --config or both --texts and --labels must be provided")
        texts = args.texts
        label_values = args.labels
        modality = args.modality
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    run_inference(
        image_path=args.input,
        output_path=args.output,
        modality=modality,
        texts=texts,
        label_values=label_values,
        inference_mode=args.mode,
        device=args.device,
        checkpoints_path=args.checkpoints,
        window_settings=window_settings,
        window_type=window_type,
        normalization_settings=normalization_settings,
        window_type_mapping=window_type_mapping,
        verbose=verbose
    )


if __name__ == '__main__':
    main()

