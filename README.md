# ESMS-Net: Enhancing Semantic-Mask Segmentation Network With Pyramid Atrousformer for Remote Sensing Image

![network](images/network.png)

## Abstract

Global context information is essential for the semantic segmentation of remote sensing (RS) images. However, most existing methods rely on a convolutional neural network (CNN), which is challenging to directly obtain the global context due to the locality of the convolution operation. Inspired by the Swin transformer with powerful global modeling capabilities, we propose a novel semantic segmentation framework for RS images called ST-U-shaped network (UNet), which embeds the Swin transformer into the classical CNN-based UNet. ST-UNet constitutes a novel dual encoder structure of the Swin transformer and CNN in parallel. First, we propose a spatial interaction module (SIM), which encodes spatial information in the Swin transformer block by establishing pixel-level correlation to enhance the feature representation ability of occluded objects. Second, we construct a feature compression module (FCM) to reduce the loss of detailed information and condense more small-scale features in patch token downsampling of the Swin transformer, which improves the segmentation accuracy of small-scale ground objects. Finally, as a bridge between dual encoders, a relational aggregation module (RAM) is designed to integrate global dependencies from the Swin transformer into the features from CNN hierarchically. Our ST-UNet brings significant improvement on the ISPRS-Vaihingen and Potsdam datasets, respectively. 

---

## Dependencies and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Wzysaber/ESMS-Net.git
cd ESMS-Net
```
### 2. Create a new conda environment
```bash
conda create -n ESMS-Net python=3.10 -y
conda activate ESMS-Net
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Datasets
Semantic Segmentation:

1. Potsdam 256x256

2. Vaihingen 256x256


## Model Training
```bash
python main.py
```

## Seg-Results
![Potsdam](images/Potsdam.png)
![Vaihingen](images/Vaihingen.png)

## Citation
```bash
@article{liu2024esms,
  title={ESMS-Net: Enhancing semantic-mask segmentation network with pyramid atrousformer for remote sensing image},
  author={Liu, Jiamin and Wang, Ziyi and Luo, Fulin and Guo, Tan and Yang, Feng and Gao, Xinbo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgement
This implementation is based on [ST-Unet](https://github.com/XinnHe/ST-UNet) . Thanks for the awesome work.
