# Medal S: Spatio-Textual Prompt Model for Medical Segmentation

**[Medal S openreview paper link](https://openreview.net/forum?id=9vCx66pnLn)**

This repository provides guidance for training and inference of Medal S within the [CVPR 2025: Foundation Models for Text-Guided 3D biomedical image segmentation](https://www.codabench.org/competitions/5651/)

## Requirements

The U-Net implementation relies on a customized version of [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures). To install it, navigate to the `model` directory and run:

```bash
cd model
pip install -e dynamic-network-architectures-main
````

**Python Version:** 3.10.16

**Key Python Packages:**

```
torch==2.2.0
transformers==4.51.3
monai==1.4.0
nibabel==5.3.2
einops
positional_encodings
scipy
pandas
scikit-learn
scikit-image
batchgenerators
acvl_utils
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

## Acknowledgements
This project is significantly improved based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/master) and [SAT](https://github.com/zhaoziheng/SAT/tree/cvpr2025challenge). We extend our gratitude to both projects.
Medal-S is developed and maintained by Medical Image Insights.
<img src="/assets/yh_logo.png" height="100px" />
