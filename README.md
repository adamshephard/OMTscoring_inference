# OMTscoring: A Fully Automated and Explainable Algorithm for the Prediction of Malignant Transformation in Oral Epithelial Dysplasia

This repository provides the inference code for the models used for predicting slide-level malignancy transformation in OED. Link to preprint [here](https://arxiv.org/abs/2307.03757). <br />

The first step in this pipeline is to use HoVer-Net+ ([see original paper](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/html/Shephard_Simultaneous_Nuclear_Instance_and_Layer_Segmentation_in_Oral_Epithelial_Dysplasia_ICCVW_2021_paper.html)) to segment the epithelium and nuclei. We have used the TIAtoolbox (see [paper here](https://www.nature.com/articles/s43856-022-00186-5)) implementation of HooVer-Net in the below scripts.

Next, we generate patch-level morphological and spatial features to use in our OMTscoring pipeline. This is currently being worked on.

After this point, these scripts require the user to have already generated patch-level (nuclear) features for use within this model. However, we will add these scripts in the near future.

## Set Up Environment

We use Python 3.10 with the [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) package installed. By default this uses PyTorch 2.0.

```
conda create -n tiatoolbox python=3.10
conda activate tiatoolbox
pip install tiatoolbox
```

## Repository Structure

Below are the main directories in the repository: 

- `dataloader/`: the data loader and augmentation pipeline
- `utils/`: scripts for metric, patch generation
- `models/`: model definition

Below are the main executable scripts in the repository:

- `run_infer.py`: main inference script for tile and WSI processing

## Inference

### Data Format
Input: <br />
- Standard images files, including `png`, `jpg` and `tiff`.
- WSIs supported by [OpenSlide](https://openslide.org/), including `svs`, `tif`, `ndpi` and `mrxs`.

Output: <br />
- ...
  
### Model Weights

Model weights obtained from training ....

If any of the above checkpoints are used, please ensure to cite the corresponding paper.

### Usage and Options

Usage: <br />
```
  run_infer.py
```
