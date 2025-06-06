import torch
import torch.nn.functional as F
from einops import repeat

def collect_fn(data):
    """
    Pad images and masks to the same depth and num of class
    
    Args:
        data : [{'text':..., 'image':..., 'mask':..., 'simulated_lowres_sc_pred':..., 'simulated_lowres_mc_pred':..., 'modality':..., 'image_path':..., 'mask_path':..., 'dataset':...}, ...]
    """
    
    image = []
    mask = []
    simulated_lowres_sc_pred = []
    simulated_lowres_mc_pred = []
    text = []
    modality = []
    image_path = []
    mask_path = []
    dataset = []

    # pad to max depth in the batch
    # pad to max num of class in the batch
    max_class = 1
    for sample in data:
        class_num = sample['mask'].shape[0]
        max_class = class_num if class_num > max_class else max_class
        
    query_mask = torch.zeros((len(data), max_class)) # bn
    for i, sample in enumerate(data):
        image.append(sample['image'])
        simulated_lowres_sc_pred.append(sample['simulated_lowres_sc_pred'])
        
        class_num = sample['mask'].shape[0]
        pad = (0, 0, 0, 0, 0, 0, 0, max_class-class_num)
        padded_mask = F.pad(sample['mask'], pad, 'constant', 0)   # nhwd
        padded_simulated_lowres_mc_pred = F.pad(sample['simulated_lowres_mc_pred'], pad, 'constant', 0)   # nhwd
        mask.append(padded_mask)
        simulated_lowres_mc_pred.append(padded_simulated_lowres_mc_pred)

        sample['text'] += ['none'] * (max_class-class_num)
        query_mask[i, :class_num] = 1.0
        
        text.append(sample['text'])
        modality.append(sample['modality'])
        image_path.append(sample['image_path'])
        mask_path.append(sample['mask_path'])
        dataset.append(sample['dataset'])

    image = torch.stack(image, dim=0)
    mask = torch.stack(mask, dim=0).float()
    simulated_lowres_sc_pred = torch.stack(simulated_lowres_sc_pred, dim=0).float()
    simulated_lowres_mc_pred = torch.stack(simulated_lowres_mc_pred, dim=0).float()
    return {'image':image, 'mask':mask, 'text':text, 'simulated_lowres_sc_pred':simulated_lowres_sc_pred, 'simulated_lowres_mc_pred':simulated_lowres_mc_pred, 'modality':modality, 'image_path':image_path, 'mask_path':mask_path, 'dataset':dataset, 'query_mask':query_mask}
