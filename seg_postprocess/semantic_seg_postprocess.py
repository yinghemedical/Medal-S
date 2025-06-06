import numpy as np
import cc3d
from typing import Union, Tuple, List, Callable, Dict

def cc3d_label_with_component_sizes(binary_image: np.ndarray, connectivity: int = 6) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Uses cc3d to label connected components in a binary image and returns the labeled image along with
    a dictionary of component sizes.

    Parameters:
    binary_image (np.ndarray): The input binary image where connected components should be labeled.
    connectivity (int): Connectivity of components (default 6 for 3D images).

    Returns:
    Tuple[np.ndarray, dict]: A tuple with the labeled image and a dictionary of component sizes.
    """
    # Ensure the image is binary
    labeled_image = cc3d.connected_components(binary_image, connectivity=connectivity)

    # Get the sizes of each component (ignoring background, i.e., label 0)
    component_sizes = {i: j for i, j in enumerate(np.bincount(labeled_image.ravel())[1:], start=1)}

    return labeled_image, component_sizes

def remove_all_but_largest_component_cc3d(binary_image: np.ndarray, connectivity: int = None) -> np.ndarray:
    """
    Removes all but the largest component in binary_image. Replaces pixels that don't belong to it with background_label
    """
    filter_fn = lambda x, y: [i for i, j in zip(x, y) if j == max(y)]
    return generic_filter_components_cc3d(binary_image, filter_fn, connectivity)


def generic_filter_components_cc3d(binary_image: np.ndarray, filter_fn: Callable[[List[int], List[int]], List[int]],
                              connectivity: int = None):
    """
    """
    labeled_image, component_sizes = cc3d_label_with_component_sizes(binary_image, connectivity)
    component_ids = list(component_sizes.keys())
    component_sizes = list(component_sizes.values())
    keep = filter_fn(component_ids, component_sizes)
    return np.isin(labeled_image.ravel(), keep).reshape(labeled_image.shape)

def remove_all_but_prob_or_size_component_cc3d(
    binary_image: np.ndarray,
    max_prob_array: np.ndarray,
    prob_threshold: float = 0.1,
    connectivity: int = 6,
) -> np.ndarray:
    """
    Removes components based on priority: max probability > size, considering only top 4 largest components.
    Special handling for close probabilities within threshold.
    
    Parameters:
    binary_image: Input binary image
    max_prob_array: Corresponding probability array (same shape as binary_image)
    prob_threshold: If max_prob - prob <= this threshold, consider probabilities "close"
    connectivity: Connectivity for connected components (default: 6 for 3D)
    
    Returns:
    Binary image with selected components kept
    """
    # Label connected components and get their sizes
    labeled_image, component_sizes = cc3d_label_with_component_sizes(binary_image, connectivity)
    
    # Handle case where there are no components
    if not component_sizes:
        return np.zeros_like(binary_image)
    
    # Get top 3 largest components
    top_components = sorted(component_sizes.items(), key=lambda x: -x[1])[:3]
    top_labels = [label for label, _ in top_components]
    
    # Calculate average probability only for top components
    component_probs = {}
    for label in top_labels:
        mask = labeled_image == label
        # Efficient mean calculation using masked array
        prob_mean = np.mean(max_prob_array[mask])
        component_probs[label] = prob_mean
        print(f"Component {label}: size={component_sizes[label]}, mean_prob={prob_mean:.4f}")
    
    # Find max probability among top 3
    max_prob = max(component_probs.values())
    
    # Find components with probabilities close to max (within threshold)
    close_prob_components = [
        label for label, prob in component_probs.items()
        if (max_prob - prob) <= prob_threshold and prob > 0.86
    ]
    
    # Case 1: Multiple components with close probabilities - keep all
    if len(close_prob_components) >= 2:
        keep = close_prob_components
        print(f"Keeping {len(keep)} components with close probabilities (threshold={prob_threshold})")
    else:
        # Case 2: Single max probability component
        max_prob_label = max(component_probs.items(), key=lambda x: x[1])[0]
        
        # Get top 2 largest among top
        top2_labels = [label for label, _ in top_components[:2]]
        
        # Case 2a: Max prob component is in top 2 size - keep it
        if max_prob_label in top2_labels: # 如果Max prob component是第二大size, 还需要满足第二大size/最大size>0.6,否则还是保留最大体积的
            keep = [max_prob_label]
            print(f"Keeping max prob component {max_prob_label}")
        # Case 2b: Otherwise keep largest component
        else:
            keep = [top_components[0][0]]
            print(f"Keeping largest component {top_components[0][0]}")
    
    # Efficiently create output binary image
    result = np.zeros_like(binary_image)
    for label in keep:
        result |= (labeled_image == label)
    
    return result.astype(binary_image.dtype)

def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask

def remove_all_but_prob_or_size_component_from_every_class_segmentation(
    segmentation: np.ndarray,
    max_prob_array: np.ndarray,
    labels_or_regions: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]],
    background_label: int = 0,
) -> np.ndarray:
    ret = np.copy(segmentation)  # do not modify the input!
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    
    for l_or_r in labels_or_regions:
        # Get mask for current label/region
        mask = region_or_label_to_mask(segmentation, l_or_r)
        # Keep only the largest component for this label/region
        mask_keep = remove_all_but_prob_or_size_component_cc3d(mask, max_prob_array, prob_threshold=0.1, connectivity=6)
        # Set non-largest components to background
        ret[mask & ~mask_keep] = background_label
    
    return ret

# def remove_all_but_largest_component_from_every_class_segmentation(
#     segmentation: np.ndarray,
#     labels_or_regions: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]],
#     background_label: int = 0,
# ) -> np.ndarray:
#     ret = np.copy(segmentation)  # do not modify the input!
#     if not isinstance(labels_or_regions, list):
#         labels_or_regions = [labels_or_regions]
    
#     for l_or_r in labels_or_regions:
#         # Get mask for current label/region
#         mask = region_or_label_to_mask(segmentation, l_or_r)
#         # Keep only the largest component for this label/region
#         mask_keep = remove_all_but_largest_component(mask)
#         # Set non-largest components to background
#         ret[mask & ~mask_keep] = background_label
    
#     return ret

