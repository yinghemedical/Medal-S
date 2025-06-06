import os
import re
import numpy as np
import math
import nibabel as nib
import torch
import random
import itertools
import json
import torch.nn.functional as F
from typing import List
from datetime import datetime
from glob import glob
from collections import Counter
from einops import rearrange
from scipy.ndimage import gaussian_filter

from data.default_resampling import resample_data_or_seg, compute_new_shape, resample_data_or_seg_to_spacing
from data.resample_torch import resample_torch_fornnunet, resample_torch_simple
from text_preprocess.extract_sentences_info import extract_info
from seg_postprocess.semantic_seg_postprocess import remove_all_but_prob_or_size_component_from_every_class_segmentation
from tqdm import tqdm
from model.medals import MedalS
from model.knowledge_encoder import Knowledge_Encoder

from scipy.ndimage import binary_fill_holes, label
from skimage.morphology import remove_small_objects
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice

def adjust_spacing_based_on_constraints(index, image_shape, current_spacing, target_spacing, crop_size, scale_factor=2.0):

    physical_constraint = (crop_size[index] * scale_factor * target_spacing[index]) / image_shape[index]
    
    if current_spacing[index] > target_spacing[index]:
        adjusted_spacing = max(target_spacing[index], physical_constraint / 1.1)
    else:
        adjusted_spacing = min(target_spacing[index], physical_constraint)

    return adjusted_spacing

def calculate_spacing_adjusted_volume(image, spacing):
    """Calculate the physical volume of the image after accounting for voxel spacing"""
    return np.prod([s * d for s, d in zip(spacing, image.shape)])

def calculate_crop_threshold(crop_size, scale_factor=2.0):
    """Calculate the threshold volume as (scale_factor * crop_size)³"""
    return (scale_factor ** 3) * np.prod(crop_size)

def adjust_spacing(img_array, img_spacing):
    img_spacing = np.asarray(img_spacing)
    min_dim_index = np.argmin(img_array.shape)
    max_spacing_index = np.argmax(img_spacing)
    
    if (min_dim_index != max_spacing_index) and (img_spacing[max_spacing_index] > 0.5):
        new_order = list(range(len(img_spacing)))
        new_order[min_dim_index], new_order[max_spacing_index] = new_order[max_spacing_index], new_order[min_dim_index]
        img_spacing = img_spacing[new_order]
    
    return img_spacing

def get_most_frequent(modality_list):
    if not modality_list:
        return "CT"
    counter = Counter(modality_list)
    return counter.most_common(1)[0][0]

def remove_small_objects_binary(binary_data, min_size=10):
    labeled_array, num_features = label(binary_data)
    sizes = np.bincount(labeled_array.ravel())
    remove = sizes < min_size
    remove[0] = False  # Ensure the background (label 0) is not removed
    labeled_array[remove[labeled_array]] = 0
    return labeled_array > 0

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
    Computes patch predictions by splitting a 3D volume into blocks of random sizes,
    processing complementary halves using a single random mask, and combining results.
    The process is repeated n times with different random masks, and results are averaged.

    Args:
        queries (torch.Tensor): Input query tensor for the model, expected shape (batch, query_dim).
        patches (torch.Tensor): Image patch tensor, expected shape (batch, channels, height, width, depth).
        lowres_single_channel_pred (torch.Tensor): Low-resolution single-channel prediction,
            shape (1, 1, h, w, d).
        lowres_multi_channel_pred (torch.Tensor): Low-resolution multi-channel prediction,
            shape (1, c, h, w, d).
        model (torch.nn.Module): Trained neural network model for prediction.
        possible_block_sizes (List[int]): List of possible block sizes (e.g., [8, 16, 32]).
        n_repeats (int, optional): Number of times to repeat prediction with different random masks.
            Defaults to 1.

    Returns:
        torch.Tensor: Averaged patch prediction from n_repeats, shape (1, c, h, w, d).

    Raises:
        ValueError: If input tensors have incorrect shapes, devices mismatch, or invalid parameters.
    """
    # Validate inputs
    if not possible_block_sizes:
        raise ValueError("possible_block_sizes cannot be empty")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    
    # Extract and validate dimensions
    if lowres_single_channel_pred.shape[0] != 1 or lowres_single_channel_pred.shape[1] != 1:
        raise ValueError("lowres_single_channel_pred must have shape (1, 1, h, w, d)")
    if lowres_multi_channel_pred.shape[0] != 1:
        raise ValueError("lowres_multi_channel_pred must have shape (1, c, h, w, d)")
    
    _, _, h, w, d = lowres_single_channel_pred.shape
    if lowres_multi_channel_pred.shape[2:] != (h, w, d):
        raise ValueError("Spatial dimensions of lowres_multi_channel_pred must match lowres_single_channel_pred")

    # Verify device consistency
    device = lowres_single_channel_pred.device
    if (queries.device != device or patches.device != device or 
        lowres_multi_channel_pred.device != device):
        raise ValueError("All input tensors must be on the same device")
    
    # Validate block sizes
    max_block_size = max(h, w, d)
    if not all(1 <= size <= max_block_size for size in possible_block_sizes):
        raise ValueError(f"Block sizes must be between 1 and {max_block_size}")

    # Initialize tensor to store sum of predictions
    prediction_sum = torch.zeros_like(lowres_multi_channel_pred, device=device)

    def upsample_block_mask(block_mask: torch.Tensor, block_size: int) -> torch.Tensor:
        """
        Upsamples a block mask to the full resolution of the input volume.

        Args:
            block_mask (torch.Tensor): Binary mask of shape (n_blocks_h, n_blocks_w, n_blocks_d).
            block_size (int): Size of each block to upsample.

        Returns:
            torch.Tensor: Upsampled mask of shape (1, 1, h, w, d).
        """
        upsampled = (
            block_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_blocks_h, n_blocks_w, n_blocks_d)
            .repeat_interleave(block_size, dim=2)  # Expand height
            .repeat_interleave(block_size, dim=3)  # Expand width
            .repeat_interleave(block_size, dim=4)  # Expand depth
            [:, :, :h, :w, :d]  # Crop to original dimensions
        ).float()
        return upsampled

    # Repeat prediction process n times
    for _ in range(n_repeats):
        # Randomly select block size
        block_size = random.choice(possible_block_sizes)

        # Compute number of blocks per dimension (ceiling division)
        n_blocks_h = (h + block_size - 1) // block_size
        n_blocks_w = (w + block_size - 1) // block_size
        n_blocks_d = (d + block_size - 1) // block_size
        total_blocks = n_blocks_h * n_blocks_w * n_blocks_d

        # Generate block mask for approximately half the blocks
        num_selected = max(1, total_blocks // 2)  # Ensure at least one block is selected
        block_mask = torch.zeros(n_blocks_h, n_blocks_w, n_blocks_d, dtype=torch.bool, device=device)
        indices = torch.randperm(total_blocks, device=device)[:num_selected]
        block_mask.view(-1)[indices] = True

        # Create full-resolution masks
        mask = upsample_block_mask(block_mask, block_size)  # Shape: (1, 1, h, w, d)
        complementary_mask = 1.0 - mask  # Complementary mask

        # Process first half: mask single- and multi-channel inputs
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

        # Process second half: mask with complementary mask
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

        # Combine predictions using complementary masks
        final_prediction = first_half_pred * complementary_mask + second_half_pred * mask

        # Add to running sum
        prediction_sum += final_prediction

    # Compute average prediction
    final_prediction = prediction_sum / n_repeats

    return final_prediction

def create_nonzero_mask(data):
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)

def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox

def respace_image(image: np.ndarray, current_spacing: np.ndarray, target_spacing: np.ndarray, device: torch.device('cpu')) -> np.ndarray:

    new_shape = compute_new_shape(image.shape[1:], current_spacing, target_spacing)
    resampled_image = resample_torch_fornnunet(
                image, new_shape, current_spacing, target_spacing,
                is_seg=False, num_threads=8, device=device,
                memefficient_seg_resampling=False,
                force_separate_z=None,
                separate_z_anisotropy_threshold=3.0
            )

    return resampled_image

def respace_mask(mask: np.ndarray, current_spacing: np.ndarray, target_spacing: np.ndarray, device: torch.device('cpu')) -> np.ndarray:

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


def read_npz_data(raw_image, raw_spacing, crop_size=[288, 288, 96],
                  target_spacing=[1.5, 1.5, 3.0], scaled_roi_lowres_pred_array=None,
                  class_name_list=[], stage_1_flag=False, device=torch.device("cuda", 0)):
    
    raw_d, raw_h, raw_w = raw_image.shape
    image = rearrange(raw_image, 'd h w -> h w d')
    spacing = raw_spacing.astype(np.float32)
    
    stage_2_spacing_1_6_descend_class_name_list = ['US_Thyroid', 'US_Jugular vein', 'US_Carotid artery', 'CT_Airway Tree', 'CT_Airways']
    stage_2_spacing_1_2_descend_class_name_list = ['US_Brain tumor']
    spacing_1_class_name_list = ["Microscopy_Endolysosomes", "Microscopy_Mitochondria"]
    spacing_descend_1_35_class_name_list = ["PET_Lesion"]

    stage_1_num_iterations_0_class_name_list = []
    stage_2_num_iterations_0_class_name_list = ['MRI_Vertebrae']
    stage_1_and_stage_2_num_iterations_0_class_name_list = ['CT_Airway Tree', 'CT_Aortic vessel trees', 'PET_Lesion', 'MRI_Spinal canal']
    num_iterations = 1

    if any(c in class_name_list for c in stage_1_and_stage_2_num_iterations_0_class_name_list):
        num_iterations = 0
    
    if stage_1_flag and any(c in class_name_list for c in stage_1_num_iterations_0_class_name_list):
        num_iterations = 0

    if not stage_1_flag and any(c in class_name_list for c in stage_2_num_iterations_0_class_name_list):
        num_iterations = 0

    max_dims = [1000, 1000, 700]
    min_dims = crop_size
    thresholds = []
    current = 1.25
    while current <= 50:
        thresholds.append(current)
        current *= 1.25
    raw_target_spacing = target_spacing.copy()
    
    for i in range(3):
        if any(c in class_name_list for c in spacing_1_class_name_list):
            spacing[i] = 1.0

        if spacing[i] < 1.0 and image.shape[i] <= max_dims[i]:
            spacing[i] = 1.0  # second stage model resolution

        if len(class_name_list) == 1 and spacing[i] > 1.0:
            if any(c in class_name_list for c in spacing_descend_1_35_class_name_list):
                if stage_1_flag:
                    spacing[i] = max(spacing[i] / 1.45, target_spacing[i])
                else:
                    spacing[i] = max(spacing[i] / 1.35, target_spacing[i])
                if spacing[i] / target_spacing[i] > 3:
                    spacing[i] = 3 * spacing[i] / (spacing[i] / target_spacing[i] - 0.5)
            else:
                spacing[i] = max(spacing[i] / 1.2, target_spacing[i])
                if spacing[i] / target_spacing[i] > 3:
                    spacing[i] = 3 * spacing[i] / (spacing[i] / target_spacing[i] - 0.5)
        
        if image.shape[i] > crop_size[i] * 2.0 and len(class_name_list) == 1:
            spacing[i] = crop_size[i] * 2.0 / image.shape[i]
        elif image.shape[i] > crop_size[i] * 1.6 and len(class_name_list) == 1 and any(c in class_name_list for c in stage_2_spacing_1_6_descend_class_name_list):
            spacing[i] = crop_size[i] * 1.6 / image.shape[i]
        elif image.shape[i] > crop_size[i] * 1.2 and len(class_name_list) == 1 and any(c in class_name_list for c in stage_2_spacing_1_2_descend_class_name_list):
            spacing[i] = crop_size[i] * 1.2 / image.shape[i]
        elif not stage_1_flag and spacing[i] < 0.4 and image.shape[i] <= max_dims[i] and image.shape[i] > crop_size[i] * 2:  # for save inference time, 2 need to test
            if len(class_name_list) == 1:
                spacing[i] = spacing[i] * 2
            else:
                spacing[i] = spacing[i] * 1.5

        if spacing[i] * image.shape[i] > max_dims[i] and spacing[i] > target_spacing[i]:
            spacing[i] = target_spacing[i]
        elif spacing[i] * image.shape[i] < min_dims[i] * target_spacing[i]:
            alpha_spacing = 1
            for threshold in reversed(thresholds):
                if image.shape[i] <= (min_dims[i] / threshold):
                    alpha_spacing = threshold
                    break

            raw_target_spacing[i] = target_spacing[i]
            target_spacing[i] = max(spacing[i] * image.shape[i] / min_dims[i], spacing[i] / alpha_spacing)
            target_spacing[i] = min(raw_target_spacing[i], target_spacing[i])

        if not stage_1_flag and any(c in class_name_list for c in ["Microscopy_Endolysosomes"]):
            spacing[i] = spacing[i] * 1.1
            
    # Calculate physical dimensions
    image_volume = calculate_spacing_adjusted_volume(image, spacing)
    scale_factor = 1.8
    crop_threshold = calculate_crop_threshold(crop_size, scale_factor=scale_factor)
    needs_downsampling = image_volume > crop_threshold
    for i in range(3):
        if needs_downsampling and spacing[i] * image.shape[i] > crop_size[i] * 1.9 * target_spacing[i] and len(class_name_list) != 1:
            spacing[i] = adjust_spacing_based_on_constraints(i, image.shape, spacing, target_spacing, crop_size, scale_factor=1.9)

    image = image[np.newaxis, ...].astype(np.float32)
    print("image.shape: ", image.shape)
    print("spacing: ", spacing)
    print("target_spacing: ", target_spacing)
    image = respace_image(image, spacing, target_spacing, torch.device('cpu'))
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
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

    def _load_model(self):
        crop_str = '_'.join(map(str, self.config['crop_size']))
        spacing_str = '_'.join(map(str, self.config['target_spacing_model']))
        
        vision_backbone_checkpoint = os.path.join(
            self.config['checkpoints_path'],
            f"nano_UNet_CVPR2025_crop_size_{crop_str}_spacing_{spacing_str}_step_{self.config['model_step']}.pth")

        model = MedalS(
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

    def run_inference(self, raw_image, raw_spacing):
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
            gaussian = torch.tensor(compute_gaussian(tuple(crop_size))).to(self.device)
            data_dict = read_npz_data(
                raw_image=raw_image,
                raw_spacing=raw_spacing,
                crop_size=crop_size,
                target_spacing=self.config['target_spacing'],
                scaled_roi_lowres_pred_array=self.config['scaled_roi_lowres_pred_array'],
                class_name_list=class_name_list,
                stage_1_flag=stage_1_flag,
                device=self.device
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
            }[modality]]).to(self.device)

            h, w, d = current_shape
            n = len(text_prompts)
            prediction = torch.zeros((n, h, w, d))
            accumulation = torch.zeros((n, h, w, d))

            with torch.autocast(self.device.type, enabled=True):
                queries = text_encoder(text_prompts, modality_code, self.device)
                torch.cuda.empty_cache()

                for patches, lowres_pred_patches, y1y2_x1x2_z1z2_ls in tqdm(
                    zip(batched_patches, batched_lowres_pred_patches if batched_lowres_pred_patches is not None else [None]*len(batched_patches), batched_y1y2_x1x2_z1z2),
                    total=len(batched_patches),
                    desc="Processing",
                    ncols=100,
                    bar_format="{l_bar}{bar:20}{r_bar}",
                    colour="green"
                ):
                    patches = patches.unsqueeze(0).to(device=self.device)

                    if not self.config['w_lowres_pred_prompts']:
                        simulated_lowres_sc_pred = torch.zeros((1, 1, *crop_size)).to(self.device)
                        simulated_lowres_mc_pred = torch.zeros((1, n, *crop_size)).to(self.device)
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
                        lowres_pred_patches = lowres_pred_patches.unsqueeze(0).to(device=self.device)
                        simulated_lowres_sc_pred = torch.where(lowres_pred_patches > 0, torch.ones_like(lowres_pred_patches), torch.zeros_like(lowres_pred_patches))
                        simulated_lowres_mc_pred = sc_mask_to_mc_mask(lowres_pred_patches, [int(i) for i in label_values])

                        possible_block_sizes = [8]
                        if instance_label == 1:
                            n_repeats = 1
                        else:
                            n_repeats = 1
                        prediction_patch = compute_patch_prediction(queries, patches, simulated_lowres_sc_pred, simulated_lowres_mc_pred, model, possible_block_sizes, n_repeats, disable_tta)

                    if instance_label == 1: # instance masks
                        for _ in range(num_iterations):  # Define num_iterations as needed
                            prediction_patch_prob = torch.sigmoid(prediction_patch).detach()
                            simulated_lowres_mc_pred = torch.where(prediction_patch_prob > 0.5, 1.0, 0.0)
                            simulated_lowres_sc_pred = (simulated_lowres_mc_pred.sum(dim=1, keepdim=True) > 0).float()  # [1, 13, 160, 160, 160] --> [1, 1, 160, 160, 160]
                            possible_block_sizes = [4]
                            n_repeats = 1
                            prediction_patch = compute_patch_prediction(queries, patches, simulated_lowres_sc_pred, simulated_lowres_mc_pred, model, possible_block_sizes, n_repeats, disable_tta)
                            

                    prediction_patch = torch.sigmoid(prediction_patch).detach()
                    y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls
                    tmp = prediction_patch[0, :, :y2-y1, :x2-x1, :z2-z1] * gaussian[:y2-y1, :x2-x1, :z2-z1]
                    prediction[:, y1:y2, x1:x2, z1:z2] += tmp.cpu()
                    accumulation[:, y1:y2, x1:x2, z1:z2] += gaussian[:y2-y1, :x2-x1, :z2-z1].cpu()

                prediction_prob = prediction / accumulation
                prediction_prob = prediction_prob.numpy()

            max_prob = np.max(prediction_prob, axis=0)
            max_class = np.argmax(prediction_prob, axis=0)
            results = np.zeros_like(max_class, dtype=np.int16)
            mask = max_prob >= 0.5
            for j, value in enumerate(label_values):
                results[(max_class == j) & mask] = int(value)
                       
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
                
                print(f"Resized segmentation from {current_h, current_w, current_d} to {original_shape}")

            pred_array = rearrange(results, 'h w d -> d h w').astype(np.int16)

            if return_max_prob and instance_label == 0:
                max_prob = remove_padding(max_prob, padding_info)
                if max_prob.shape != original_shape:
                    max_prob = resample_torch_simple(
                        max_prob[np.newaxis, ...],
                        new_shape=original_shape,
                        is_seg=False,
                        num_threads=4,
                        device=torch.device('cpu'),
                        memefficient_seg_resampling=False).squeeze(0)

                    print(f"Resized max probability from {current_h, current_w, current_d} to {original_shape}")
                max_prob = rearrange(max_prob, 'h w d -> d h w').astype(np.float32)
        
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
):
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
    }
    
    pipeline = MedicalSegmentationPipeline(config)
    return pipeline.run_inference(raw_image, raw_spacing)

def main():
    # set gpu
    device = torch.device("cuda", 0)

    stage_1_crop_size = [224, 224, 128]
    stage_1_target_spacing = [1.5, 1.5, 3.0]
    stage_1_target_spacing_model = [1.5, 1.5, 3.0]
    stage_1_model_step = 358600
    stage_1_disable_tta = True

    stage_2_crop_size = [192, 192, 192]
    stage_2_target_spacing = [1.0, 1.0, 1.0]
    stage_2_target_spacing_model = [1.0, 1.0, 1.0]
    stage_2_model_step = 341300
    stage_2_disable_tta = True # always True

    npz_file_path = glob("./inputs/*.npz")[0]
    file_name = os.path.basename(npz_file_path)
    output_folder = "./outputs"
    
    # Load original image
    data = np.load(npz_file_path, allow_pickle=True)
    img_array = data['imgs']
    img_spacing = data['spacing']
    text_prompts = data['text_prompts']
    instance_label = data['text_prompts'].item()['instance_label']
    print("text_prompts: ", text_prompts)
    texts = [value for key, value in text_prompts.item().items() if key != 'instance_label']
    label_values = [key for key, value in text_prompts.item().items() if key != 'instance_label']

    print("img_array.shape: ", img_array.shape)
    print("img_spacing: ", img_spacing)

    # Load class_mapping.json
    with open('./text_preprocess/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)

    with open('./text_preprocess/variant_mapping.json', 'r') as f:
        variant_mapping = json.load(f)

    # Define modality keywords
    modality_keywords = {
        'Microscopy': ['microscopy', 'microscope', 'Microscope', 'Microscopy', 'microscopic', 'Microscopic', 'microscopical', 'Microscopical', 'ultrastructural', 'Ultrastructural', 'ultrastructure', 'Ultrastructure', 'EM', 'light sheet', 'Light sheet'],
        'PET': ['PET', 'positron emission tomography'],
        'US': ['US', 'ultrasound', 'Ultrasound', 'Echocardiography', 'echocardiography', 'Echocardiographic', 'echocardiographic', 'ultrasonic', 'Ultrasonic'],
        'MRI': ['MR', 'MRI', 'magnetic resonance', 'Magnetic resonance', 'Magnetic Resonance', 'diffusion', 'Diffusion', 'DWI', 'ADC', 'pelvic', 'FLAIR'],
        'CT': ['CT', 'computed tomography', 'Computed tomography', 'Computed Tomography', 'tomographic', 'Tomographic', 'cross-sectional']
    }
    direction_patterns = {
        'left': re.compile(r'\b(left)\b', re.IGNORECASE),
        'right': re.compile(r'\b(right)\b', re.IGNORECASE)
    }
    modality_list = []
    raw_class_name_list = []
    for sentence in texts:
        # print("sentence, instance_label: ", sentence, instance_label)
        result = extract_info(sentence, instance_label, class_mapping, variant_mapping, modality_keywords, direction_patterns)
        # print("result: ", result)
        modality = result['modality']
        class_name = result['class']['name']
        modality_list.append(modality)
        raw_class_name_list.append(modality + "_" + class_name)
    
    class_name_list = list(set(raw_class_name_list))
    
    modality = get_most_frequent(modality_list).lower()
    print("modality: ", modality)

    img_spacing_transposed = img_spacing[[2, 0, 1]]

    for i in range(3):
        if img_spacing_transposed[i] < 0.1:
            img_spacing_transposed[i] = 1.0 # # second stage model resolution

    img_spacing_transposed = adjust_spacing(img_array, img_spacing_transposed).astype(np.float32)
    print("adjust img_spacing_transposed: ", img_spacing_transposed)
    
    img_spacing = img_spacing_transposed[[1, 2, 0]]
    print("modified img_spacing: ", img_spacing)

    skip_stage_1 = False
    skip_stage_2 = False

    lowres_class_name_list = ["US_Soleus", "US_Gastrocnemius Medialis", "US_Gastrocnemius Lateralis"]
    highres_class_name_list = ["Microscopy_Synaptic clefts"]

    if any(c in class_name_list for c in lowres_class_name_list):
        skip_stage_1 = False
        skip_stage_2 = True
    elif any(c in class_name_list for c in highres_class_name_list):
        skip_stage_1 = True
        skip_stage_2 = False

    with open('./seg_postprocess/improved_classes.json', 'r') as f:
        improved_classes_data = json.load(f)

    postprocess_class_name_list = []

    for modality_improved, classes_improved in improved_classes_data.items():
        if 'anatomy' in classes_improved:
            for c_improved in classes_improved['anatomy'].keys():
                postprocess_class_name_list.append(f"{modality_improved}_{c_improved}")

    # print("postprocess_class_name_list: ", postprocess_class_name_list)

    if any(c in class_name_list for c in postprocess_class_name_list):
        return_max_prob = True
    else:
        return_max_prob = False

    final_pred_array = np.zeros_like(img_array, dtype=np.int16)
    
    if not skip_stage_1:
        # Stage 1: Low-resolution segmentation
        lowres_pred_array, lowres_max_prob_array = run_segmentation(
            raw_image=img_array,
            raw_spacing=img_spacing,
            crop_size=stage_1_crop_size,
            target_spacing=stage_1_target_spacing,
            target_spacing_model=stage_1_target_spacing_model,
            w_lowres_pred_prompts=False,
            scaled_roi_lowres_pred_array=None,
            disable_tta=stage_1_disable_tta, #False,
            model_step=stage_1_model_step,
            modality=modality,
            instance_label=instance_label,
            texts=texts,
            label_values=label_values,
            return_max_prob=return_max_prob,
            class_name_list=class_name_list,
            stage_1_flag=True,
            device=device
        )
        if lowres_pred_array.sum() == 0:
            skip_stage_1 = True

    if skip_stage_2:
        final_pred_array = lowres_pred_array.astype(np.int16)
        if return_max_prob and instance_label == 0:
            final_max_prob_array = lowres_max_prob_array.astype(np.float32)
    else:
        if instance_label == 0:
            min_size = 10
            lowres_pred_binary_array = (lowres_pred_array > 0).astype(np.int16)
            # Remove small objects from binary mask
            lowres_pred_binary_array = remove_small_objects_binary(lowres_pred_binary_array, min_size=min_size).astype(np.int16)
            lowres_pred_array = lowres_pred_array * lowres_pred_binary_array

        if skip_stage_1:
            print("Skip low-resolution prediction. Proceeding to Stage 2.")
            # Stage 2: Perform high-resolution segmentation on the entire image

            final_pred_array, final_max_prob_array = run_segmentation(
                raw_image=img_array,
                raw_spacing=img_spacing,
                crop_size=stage_2_crop_size,
                target_spacing=stage_2_target_spacing, #[0.75, 0.75, 0.75], #[1.0, 1.0, 1.0],
                target_spacing_model=stage_2_target_spacing_model, #[0.75, 0.75, 0.75], #[1.0, 1.0, 1.0],
                w_lowres_pred_prompts=False,
                scaled_roi_lowres_pred_array=None,
                disable_tta=True,
                model_step=stage_2_model_step,
                modality=modality,
                instance_label=instance_label,
                texts=texts,
                label_values=label_values,
                return_max_prob=return_max_prob,
                class_name_list=class_name_list,
                stage_1_flag=False,
                device=device
            )
        else:
            # Compute ROI using non-binarized prediction
            m_1_5_class_name_list = ['CT_Airway Tree']
            m_1_2_class_name_list = [] #['CT_Aortic vessel trees']
            if any(c in class_name_list for c in m_1_5_class_name_list):
                m = 1.5
            elif any(c in class_name_list for c in m_1_2_class_name_list):
                m = 1.2
            else:
                m = 1.1  # Scaling factor for ROI
            print(" Scaling factor for ROI m: ", m)
            non_zero_indices = np.argwhere(lowres_pred_array > 0)
            
            x_min, y_min, z_min = non_zero_indices.min(axis=0)
            x_max, y_max, z_max = non_zero_indices.max(axis=0)
            
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2
            
            x_range = (x_max - x_min + 1) * m / 2
            y_range = (y_max - y_min + 1) * m / 2
            z_range = (z_max - z_min + 1) * m / 2

            spacing_descend_1_35_class_name_list = ["PET_Lesion"]
            spacing_1_class_name_list = ["Microscopy_Endolysosomes", "Microscopy_Mitochondria"]
            stage_2_spacing_1_6_descend_class_name_list = ['US_Thyroid', 'US_Jugular vein', 'US_Carotid artery', 'CT_Airway Tree']
            stage_2_spacing_1_2_descend_class_name_list = ['US_Brain tumor']
            
            print("Initial ranges - x_range, y_range, z_range:", x_range, y_range, z_range)
            # Compute minimum ranges based on stage_2_crop_size and spacing
            max_dims = [700, 1000, 1000]

            for i in range(3):
                if any(c in class_name_list for c in spacing_1_class_name_list):
                    img_spacing_transposed[i] = 1.0
                
                if img_spacing_transposed[i] < 1.0 and img_array.shape[i] <= max_dims[i]:
                    img_spacing_transposed[i] = 1.0

                if len(class_name_list) == 1 and img_spacing_transposed[i] > 1.0:
                    if any(c in class_name_list for c in spacing_descend_1_35_class_name_list):
                        img_spacing_transposed[i] = max(img_spacing_transposed[i] / 1.35, stage_2_target_spacing[i])
                        if img_spacing_transposed[i] / stage_2_target_spacing[i] > 3:
                            img_spacing_transposed[i] = 3 * img_spacing_transposed[i] / (img_spacing_transposed[i] / stage_2_target_spacing[i] - 0.5)
                    else:
                        img_spacing_transposed[i] = max(img_spacing_transposed[i] / 1.2, stage_2_target_spacing[i])
                        if img_spacing_transposed[i] / stage_2_target_spacing[i] > 3:
                            img_spacing_transposed[i] = 3 * img_spacing_transposed[i] / (img_spacing_transposed[i] / stage_2_target_spacing[i] - 0.5)
                
                if img_array.shape[i] > stage_2_crop_size[i] * 2.0 and len(class_name_list) == 1:
                    img_spacing_transposed[i] = stage_2_crop_size[i] * 2.0 / img_array.shape[i]
                elif img_array.shape[i] > stage_2_crop_size[i] * 1.6 and len(class_name_list) == 1 and any(c in class_name_list for c in stage_2_spacing_1_6_descend_class_name_list):
                    img_spacing_transposed[i] = stage_2_crop_size[i] * 1.6 / img_array.shape[i]
                elif img_array.shape[i] > stage_2_crop_size[i] * 1.2 and len(class_name_list) == 1 and any(c in class_name_list for c in stage_2_spacing_1_2_descend_class_name_list):
                    img_spacing_transposed[i] = stage_2_crop_size[i] * 1.2 / img_array.shape[i]
                elif img_spacing_transposed[i] < 0.4 and img_array.shape[i] <= max_dims[i] and img_array.shape[i] > stage_2_crop_size[i] * 2: # for save inference time, 2 need to test
                    if len(class_name_list) == 1:
                        img_spacing_transposed[i] = img_spacing_transposed[i] * 2
                    else:
                        img_spacing_transposed[i] = img_spacing_transposed[i] * 1.5
                
                if img_spacing_transposed[i] * img_array.shape[i] > max_dims[i] and img_spacing_transposed[i] > stage_2_target_spacing[i]:
                    img_spacing_transposed[i] = stage_2_target_spacing[i]
                
                if any(c in class_name_list for c in ["Microscopy_Endolysosomes"]):
                    img_spacing_transposed[i] = img_spacing_transposed[i] * 1.1
            
             # Calculate physical dimensions
            image_volume = calculate_spacing_adjusted_volume(img_array, img_spacing_transposed)
            # Compute threshold (2x crop size in each dimension = 8x volume)
            crop_threshold = calculate_crop_threshold(stage_2_crop_size, scale_factor=1.8)
            needs_downsampling = image_volume > crop_threshold
            for i in range(3):
                if needs_downsampling and img_spacing_transposed[i] * img_array.shape[i] > stage_2_crop_size[i] * 1.9 * stage_2_target_spacing[i] and len(class_name_list) != 1:
                    img_spacing_transposed[i] = adjust_spacing_based_on_constraints(i, img_array.shape, img_spacing_transposed, stage_2_target_spacing, stage_2_crop_size, scale_factor=1.9)

            min_x_range = (stage_2_crop_size[0] / 2) * stage_2_target_spacing[0] / img_spacing_transposed[0]
            min_y_range = (stage_2_crop_size[1] / 2) * stage_2_target_spacing[1] / img_spacing_transposed[1]
            min_z_range = (stage_2_crop_size[2] / 2) * stage_2_target_spacing[2] / img_spacing_transposed[2]

            # Adjust ranges to be at least the minimum required, and ensure they are integers
            x_range = max(min_x_range-1, x_range)
            y_range = max(min_y_range-1, y_range)
            z_range = max(min_z_range-1, z_range)

            print("Adjusted ranges - x_range, y_range, z_range:", x_range, y_range, z_range)
            
            x_min_new = max(0, int(x_center - x_range))
            x_max_new = min(lowres_pred_array.shape[0] - 1, int(x_center + x_range))
            y_min_new = max(0, int(y_center - y_range))
            y_max_new = min(lowres_pred_array.shape[1] - 1, int(y_center + y_range))
            z_min_new = max(0, int(z_center - z_range))
            z_max_new = min(lowres_pred_array.shape[2] - 1, int(z_center + z_range))
            
            print("img_array.shape: ", img_array.shape)
            print("lowres_pred_array.shape: ", lowres_pred_array.shape)
            # Extract ROI image and ground truth
            scaled_roi_array = img_array[x_min_new:x_max_new+1, y_min_new:y_max_new+1, z_min_new:z_max_new+1]
            scaled_roi_lowres_pred_array = lowres_pred_array[x_min_new:x_max_new+1, y_min_new:y_max_new+1, z_min_new:z_max_new+1]
            print("scaled_roi_array.shape: ", scaled_roi_array.shape)
            
            w_lowres_pred_prompts = True
            # Stage 2: High-resolution segmentation on ROI
            roi_pred_array, roi_max_prob_array = run_segmentation(
                raw_image=scaled_roi_array,
                raw_spacing=img_spacing,
                crop_size=stage_2_crop_size,
                target_spacing=stage_2_target_spacing, #[0.75, 0.75, 0.75], #[1.0, 1.0, 1.0],
                target_spacing_model=stage_2_target_spacing_model, #[0.75, 0.75, 0.75], #[1.0, 1.0, 1.0],
                w_lowres_pred_prompts=w_lowres_pred_prompts,
                scaled_roi_lowres_pred_array=scaled_roi_lowres_pred_array,
                disable_tta=stage_2_disable_tta,
                model_step=stage_2_model_step,
                modality=modality,
                instance_label=instance_label,
                texts=texts,
                label_values=label_values,
                return_max_prob=return_max_prob,
                class_name_list=class_name_list,
                stage_1_flag=False,
                device=device
            )

            # Integrate ROI prediction back into full volume
            final_pred_array = np.zeros_like(lowres_pred_array, dtype=np.int16)
            final_pred_array[x_min_new:x_max_new+1, y_min_new:y_max_new+1, z_min_new:z_max_new+1] = roi_pred_array
            if return_max_prob and instance_label == 0:
                final_max_prob_array = np.zeros_like(lowres_pred_array, dtype=np.float32)
                final_max_prob_array[x_min_new:x_max_new+1, y_min_new:y_max_new+1, z_min_new:z_max_new+1] = roi_max_prob_array

    if instance_label == 1:
        min_size = 10
        final_pred_array = remove_small_objects_binary(final_pred_array, min_size=min_size).astype(np.int16)

    x, y, z = final_pred_array.shape

    # Postprocess:
    if return_max_prob and instance_label == 0 and x * y * z < 650 * 650 * 100:
        label_values_postprocess = []
        for i in range(0, len(raw_class_name_list)):
            class_name = raw_class_name_list[i]
            if class_name in postprocess_class_name_list:
                label_value = int(label_values[i])
                label_values_postprocess.append(label_value)
        
        label_values_postprocess = list(set(label_values_postprocess))
        print("label_values_postprocess: ", label_values_postprocess)
        final_pred_array = remove_all_but_prob_or_size_component_from_every_class_segmentation(final_pred_array, final_max_prob_array, label_values_postprocess, background_label=0)

    # Save final prediction
    np.savez_compressed(os.path.join(output_folder, file_name), segs=final_pred_array, spacing=img_spacing, text_prompts=text_prompts)
    # # Save max probability
    # np.savez_compressed(os.path.join(output_folder, file_name.replace(".npz", "_max_prob.npz")), max_prob=final_max_prob_array, spacing=img_spacing, text_prompts=text_prompts)

if __name__ == '__main__':
    main()