# Medal S: Spatio-Textual Prompt Model for Medical Segmentation

[![Paper](https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg)](https://arxiv.org/abs/2511.13001)
[![OpenReview](https://img.shields.io/badge/OpenReview-Discussion-4CAF50.svg)](https://openreview.net/forum?id=9vCx66pnLn#discussion)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow.svg)](https://huggingface.co/spc819/Medal-S-V1.0)

This repository provides guidance for training and inference of Medal S within the [CVPR 2025: Foundation Models for Text-Guided 3D biomedical image segmentation](https://www.codabench.org/competitions/5651/)

Docker link for the 2025/05/30 testing submission: [Medal S](https://drive.google.com/file/d/1HRJqYUXajptGsKaXEhn-s3rGcnKIwGs7/view)

## Requirements

**Python Version:** 3.10.16

### Installation

1. **Create conda environment and install dependencies:**

```bash
#!/bin/bash

# Create environment
conda create -n medals_local_test python=3.10 -y

conda activate medals_local_test

# Install packages
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3 monai==1.4.0 nibabel==5.3.2 tensorboard einops positional_encodings scipy pandas scikit-learn scikit-image batchgenerators acvl_utils

# Install nnU-Net
wget https://github.com/MIC-DKFZ/nnUNet/archive/refs/tags/v2.4.1.tar.gz
tar -xvf v2.4.1.tar.gz
pip install -e nnUNet-2.4.1

# Install dynamic network architectures
cd model && pip install -e dynamic-network-architectures-main && cd ..
```

## Training Guidance

First, download the dataset from [Hugging Face: junma/CVPR-BiomedSegFM](https://huggingface.co/datasets/junma/CVPR-BiomedSegFM).

* **Data Preparation**: Preprocess and organize all training data into a `train_all.jsonl` file using the provided script: `data/challenge_data/get_train_jsonl.py`.

* **Knowledge Enhancement**: You can either use the pre-trained text encoder from SAT ([https://github.com/zhaoziheng/SAT/tree/cvpr2025challenge](https://github.com/zhaoziheng/SAT/tree/cvpr2025challenge)) available on [Hugging Face](https://huggingface.co/zzh99/SAT/tree/main/Pretrain), or pre-train it yourself following the guidance in this [repository](https://github.com/zhaoziheng/SAT-Pretrain/tree/master). As recommended by SAT, we **freeze** the text encoder when training the segmentation model.

* **Segmentation**: The training script is located at `sh/cvpr2025_Blosc2_pretrain_1.0_1.0_1.0_UNET_ps192.sh`. Before training, NPZ files will be converted to the Blosc2 compressed format (from the nnU-Net framework).

Training takes approximately 7 days with 2x H100-80GB GPUs for a 224x224x128 (1.5, 1.5, 3.0) spacing model, using a batch size of 2 per GPU. For a 192x192x192 (1.0, 1.0, 1.0) spacing model, it requires 4x H100-80GB GPUs with a batch size of 2 per GPU. You may modify the patch size and batch size to train on GPUs with less memory.

## Inference Guidance
We provide inference code for test data:

```bash
python inference.py
```

## Citation
```
@misc{shi2025medalsspatiotextualprompt,
      title={Medal S: Spatio-Textual Prompt Model for Medical Segmentation}, 
      author={Pengcheng Shi and Jiawei Chen and Jiaqi Liu and Xinglin Zhang and Tao Chen and Lei Li},
      year={2025},
      eprint={2511.13001},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.13001}, 
}
```

## Acknowledgements
This project is significantly improved based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/master) and [SAT](https://github.com/zhaoziheng/SAT/tree/cvpr2025challenge). We extend our gratitude to both projects.
Medal-S is developed and maintained by Medical Image Insights.
<img src="/assets/yh_logo.png" height="100px" />
