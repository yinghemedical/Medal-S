import os
import json
import numpy as np
import torch
import argparse
from tqdm import tqdm
import nibabel as nib
import warnings
import signal
import pandas as pd
import math
import time 
import random
from datetime import datetime
from typing import Tuple, Union, List
from einops import rearrange
from multiprocessing import Pool

from default_resampling import resample_data_or_seg, compute_new_shape, resample_data_or_seg_to_spacing
from resample_torch import resample_torch_fornnunet
from nnunet_dataset import nnUNetDatasetBlosc2
import numpy as np
from scipy.ndimage import binary_fill_holes
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice

def adjust_spacing(img_array, img_spacing):
    img_spacing = np.asarray(img_spacing)
    min_dim_index = np.argmin(img_array.shape)
    max_spacing_index = np.argmax(img_spacing)
    
    if (min_dim_index != max_spacing_index) and (img_spacing[max_spacing_index] > 0.5):
        new_order = list(range(len(img_spacing)))
        new_order[min_dim_index], new_order[max_spacing_index] = new_order[max_spacing_index], new_order[min_dim_index]
        img_spacing = img_spacing[new_order]
    
    return img_spacing

def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)

def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
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
    """
    Resample image to target spacing using linear interpolation.
    """

    # resampled_image = resample_data_or_seg_to_spacing(
    #             image, current_spacing, target_spacing,
    #             is_seg=False, order=3, order_z=0,
    #             force_separate_z=None,
    #             separate_z_anisotropy_threshold=3.0
    #         )
    
    new_shape = compute_new_shape(image.shape[1:], current_spacing, target_spacing)
    resampled_image = resample_torch_fornnunet(
                image, new_shape, current_spacing, target_spacing,
                is_seg=False, num_threads=4, device=device,
                memefficient_seg_resampling=False,
                force_separate_z=None,
                separate_z_anisotropy_threshold=3.0
            )

    return resampled_image

def respace_mask(mask: np.ndarray, current_spacing: np.ndarray, target_spacing: np.ndarray, device: torch.device('cpu')) -> np.ndarray:
    """
    Resample mask to target spacing using nearest-neighbor interpolation.
    """
    # resampled_mask = resample_data_or_seg_to_spacing(mask,
    #                             current_spacing,
    #                             target_spacing,
    #                             is_seg=True,
    #                             order=1, order_z=0,
    #                             force_separate_z=None,
    #                             separate_z_anisotropy_threshold=3.0)

    new_shape = compute_new_shape(mask.shape[1:], current_spacing, target_spacing)
    resampled_mask = resample_torch_fornnunet(
                mask, new_shape, current_spacing, target_spacing,
                is_seg=True, num_threads=4, device=device,
                memefficient_seg_resampling=False,
                force_separate_z=None,
                separate_z_anisotropy_threshold=3.0
            )

    return resampled_mask

def sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
    num_samples = 10000
    min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
    # sparse
    rndst = np.random.RandomState(seed)
    class_locs = {}
    foreground_mask = seg != 0
    foreground_coords = np.argwhere(foreground_mask)
    seg = seg[foreground_mask]
    del foreground_mask
    unique_labels = pd.unique(seg.ravel())

    # We don't need more than 1e7 foreground samples. That's insanity. Cap here
    if len(foreground_coords) > 1e7:
        take_every = math.floor(len(foreground_coords) / 1e7)
        # keep computation time reasonable
        if verbose:
            print(f'Subsampling foreground pixels 1:{take_every} for computational reasons')
        foreground_coords = foreground_coords[::take_every]
        seg = seg[::take_every]

    for c in classes_or_regions:
        k = c if not isinstance(c, list) else tuple(c)

        # check if any of the labels are in seg, if not skip c
        if isinstance(c, (tuple, list)):
            if not any([ci in unique_labels for ci in c]):
                class_locs[k] = []
                continue
        else:
            if c not in unique_labels:
                class_locs[k] = []
                continue

        if isinstance(c, (tuple, list)):
            mask = seg == c[0]
            for cc in c[1:]:
                mask = mask | (seg == cc)
            all_locs = foreground_coords[mask]
        else:
            mask = seg == c
            all_locs = foreground_coords[mask]
        if len(all_locs) == 0:
            class_locs[k] = []
            continue
        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

        selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
        class_locs[k] = selected
        if verbose:
            print(c, target_num_samples)
        seg = seg[~mask]
        foreground_coords = foreground_coords[~mask]
    return class_locs

def save_as_nifti(data: np.ndarray, spacing: list, output_path: str):
    """Save numpy array as NIfTI file with given spacing"""
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1])
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_path)

def process_file(args):
    """
    处理单个.npz文件
    """
    input_path, output_path, target_spacing, crop_size, device = args
    challenge_data_dir = "/data/shipengcheng/code/CVPR2025_Text_guided_seg/SAT_cvpr2025challenge/data/challenge_data"
    text_prompts_json = challenge_data_dir + "/CVPR25_TextSegFMData_with_class.json"
    
    # Load text prompts
    with open(text_prompts_json, 'r') as f:
        text_prompts = json.load(f)
    
    dataset = os.path.basename(os.path.dirname(input_path))

    classes_list = list(text_prompts[dataset].keys())
    
    if "instance_label" in classes_list:
        classes_list.remove("instance_label")
    
    foreground_regions = [int(i) for i in classes_list]

    # 加载数据
    data = np.load(input_path)
    img = data['imgs']  # 0~255
    sc_mask = data['gts']
    spacing = data['spacing'].tolist()
    
    # 重排维度
    img = rearrange(img, 'd h w -> h w d')
    sc_mask = rearrange(sc_mask, 'd h w -> h w d')

    # spacing:
    for i in range(3):
        if spacing[i] <= 0.1:
            spacing[i] = 1.0

    spacing = adjust_spacing(img, spacing)

    max_dims = [1000, 1000, 700]
    min_dims = crop_size
    thresholds = []
    current = 1.25
    while current <= 50:
        thresholds.append(current)
        current *= 1.25
    raw_target_spacing = target_spacing.copy()
    for i in range(3):
        if dataset == "Microscopy_urocell_Endolysosomes" or dataset == "Microscopy_urocell_Mitochondria":
            spacing[i] = 1.0

        if spacing[i] < 1.0 and img.shape[i] <= max_dims[i]:
            spacing[i] = 1.0 # second stage model resolution
            
        # if spacing[i] * img.shape[i] > max_dims[i] * target_spacing[i]: 
        if spacing[i] * img.shape[i] > max_dims[i] * target_spacing[i] and spacing[i] > target_spacing[i]:
            spacing[i] = target_spacing[i]
        elif spacing[i] * img.shape[i] < min_dims[i] * target_spacing[i]:
            alpha_spacing = 1
            for threshold in reversed(thresholds):
                if img.shape[i] <= (min_dims[i] / threshold):
                    alpha_spacing = threshold
                    break

            raw_target_spacing[i] = target_spacing[i]
            target_spacing[i] = max(spacing[i] * img.shape[i] / min_dims[i], spacing[i] / alpha_spacing)
            print("alpha_spacing: ", alpha_spacing)
            print("spacing[i] * img.shape[i] / min_dims[i], spacing[i] / alpha_spacing: ", spacing[i] * img.shape[i] / min_dims[i], spacing[i] / alpha_spacing)
            print("raw_target_spacing[i], target_spacing[i]: ", raw_target_spacing[i], target_spacing[i])
            target_spacing[i] = min(raw_target_spacing[i], target_spacing[i])
            print("dataset, img.shape[i], min_dims[i], target_spacing[i], spacing[i]): ", dataset, img.shape[i], min_dims[i], target_spacing[i], spacing[i])
    
    img = img[np.newaxis, ...].astype(np.float32)
    sc_mask = sc_mask[np.newaxis, ...].astype(np.int16)
    
    assert img.shape[1:] == sc_mask.shape[1:], "Shape mismatch between image and segmentation."
    sc_mask = np.copy(sc_mask)

    properties = {}
    properties['spacing'] = spacing

    # crop, remember to store size before cropping!
    shape_before_cropping = img.shape[1:]
    properties['shape_before_cropping'] = shape_before_cropping
    # this command will generate a segmentation. This is important because of the nonzero mask which we may need
    img, sc_mask, bbox = crop_to_nonzero(img, sc_mask)
    properties['bbox_used_for_cropping'] = bbox
    properties['shape_after_cropping_and_before_resampling'] = img.shape[1:]

    # resample image and segmentation mask to target spacing
    if device == torch.device('cpu'):
        img_resampled = respace_image(img, spacing, target_spacing, device)
        sc_mask_resampled = respace_mask(sc_mask, spacing, target_spacing, device)
    else:
        with torch.no_grad():
            img_tensor = torch.from_numpy(img).to(device)
            sc_mask_tensor = torch.from_numpy(sc_mask).to(device)
            img_resampled = respace_image(img_tensor, spacing, target_spacing, device)
            sc_mask_resampled = respace_mask(sc_mask_tensor, spacing, target_spacing, device)
            img_resampled = img_resampled.cpu().numpy()
            sc_mask_resampled = sc_mask_resampled.cpu().numpy()

        del img_tensor, sc_mask_tensor
        torch.cuda.empty_cache()

    # class_locations:
    collect_for_this = foreground_regions

    # no need to filter background in regions because it is already filtered in handle_labels
    # print(all_labels, regions)
    properties['class_locations'] = sample_foreground_locations(sc_mask_resampled, collect_for_this, verbose=False)
    if np.max(sc_mask_resampled) > 127:
        sc_mask_resampled = sc_mask_resampled.astype(np.int16)
    else:
        sc_mask_resampled = sc_mask_resampled.astype(np.int8)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img_resampled = img_resampled.astype(np.float32, copy=False)
    sc_mask_resampled = sc_mask_resampled.astype(np.int16, copy=False)

    assert img_resampled.shape == sc_mask_resampled.shape, "Shape mismatch between image and segmentation after resampling."

    # print('dtypes', img_resampled.dtype, sc_mask_resampled.dtype)
    block_size_data, chunk_size_data = nnUNetDatasetBlosc2.comp_blosc2_params(
        img_resampled.shape,
        tuple(crop_size),
        img_resampled.itemsize)
    block_size_seg, chunk_size_seg = nnUNetDatasetBlosc2.comp_blosc2_params(
        sc_mask_resampled.shape,
        tuple(crop_size),
        sc_mask_resampled.itemsize)

    nnUNetDatasetBlosc2.save_case(img_resampled, sc_mask_resampled, properties, output_path[:-4],
                                    chunks=chunk_size_data, blocks=block_size_data,
                                    chunks_seg=chunk_size_seg, blocks_seg=block_size_seg)

def preprocess_dataset(jsonl_path, target_spacing, crop_size, num_workers=8):
    """
    预处理整个数据集
    """
    # 收集所有需要处理的文件
    tasks = []
    
    # 读取jsonl文件
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    x, y, z = target_spacing
    ps_x, ps_y, ps_z = crop_size
    target_spacing_folder_name = "3D_train_npz_all_spacing_xX_yX_zX_ps_A_B_C".replace('xX', f'x{x}').replace('yX', f'y{y}').replace('zX', f'z{z}').replace('A', f'{ps_x}').replace('B', f'{ps_y}').replace('C', f'{ps_z}')
    
    for line in lines:
        data_info = json.loads(line)
        input_path = data_info['data']
        
        # 构建输出路径
        output_path = input_path.replace(
            'CVPR-BiomedSegFM/3D_train_npz_all', 
            'CVPR-BiomedSegFM_preprocess/' + target_spacing_folder_name
        )

        output_img_blosc2_path = output_path[:-4] + '.b2nd'
        output_seg_blosc2_path = output_path[:-4] + '_seg.b2nd'
        output_pkl_path = output_path[:-4] + '.pkl'

        if os.path.exists(output_img_blosc2_path) and os.path.exists(output_seg_blosc2_path) and os.path.exists(output_pkl_path):
            continue
        
        devices = ['cpu', 'cuda:0', 'cuda:1']
        weights = [0.6, 0.2, 0.2]
        selected_device = random.choices(devices, weights=weights, k=1)[0]
        device = torch.device(selected_device)

        tasks.append((input_path, output_path, target_spacing, crop_size, device))
    
    print(f"Found {len(tasks)} files to process")
    
    # 使用多进程处理，并添加进度条
    with Pool(num_workers) as pool:
        # 使用tqdm显示进度
        for _ in tqdm(pool.imap(process_file, tasks), total=len(tasks), desc="Processing files"):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_jsonl', type=str, 
                       default='/data/shipengcheng/code/CVPR2025_Text_guided_seg/SAT_cvpr2025challenge/data/challenge_data/train_all.jsonl',
                       help='Path to jsonl file containing dataset information')
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.5, 1.5, 3.0],
                       help='Target spacing in format x y z')
    parser.add_argument('--crop_size', type=int, nargs=3, default=[192, 192, 96],
                       help='Crop size for Blosc2 in format x y z')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of workers for parallel processing')
    
    args = parser.parse_args()
    
    print(f"JSONL file: {args.datasets_jsonl}")
    print(f"Target spacing: {args.target_spacing}")
    print(f"Crop size for Blosc2: {args.crop_size}")
    print(f"Using {args.num_workers} workers")
    
    preprocess_dataset(args.datasets_jsonl, args.target_spacing, args.crop_size, args.num_workers)