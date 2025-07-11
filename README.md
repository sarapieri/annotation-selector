# Annotation Selector

A graphical user interface (GUI) tool designed for viewing, navigating, and selecting images from panoptic segmentation datasets. It provides a set of features to streamline the process of curating image subsets for computer vision tasks.

## 🛠️ TODO

- [ ] Check installation instructions (below raw version)
- [ ] Add usage instructions
- [ ] Add screenshots or demo GIF of the GUI tool


## Environment Setup

Follow these steps to get started:

### 1. Create a Conda Environment

```bash
conda create -n ann_sel python=3.10 -y
conda activate ann_sel
```

### 2. Install [Detectron2](https://github.com/facebookresearch/detectron2)

### 3. Install Other Dependencies


```bash
conda install numpy matplotlib pillow -y
pip install tqdm
pip install PyQt6
pip install git+https://github.com/cocodataset/panopticapi.git
pip install opencv-python-headless
```
## 📁 Dataset Preparation (VIPSeg)

This tool is designed to work with the [VIPSeg Dataset](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset).

### 1. Download the Dataset

Follow the official instructions to download the VIPSeg dataset from the [VIPSeg GitHub Repository](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset).

### 2. Configure Dataset Paths

Create a `config.json` file in the root directory of the project to define paths to the dataset files.

Example `config.json`:

```json
{
  "VIPSeg": {
    "image_dir": "/path/to/VIPSeg/VIPSeg_720P/images",
    "mask_dir": "/path/to/VIPSeg/VIPSeg_720P/panomasksRGB",
    "ann_file": "/path/to/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json"
  }
}

