# OMTscoring: A Fully Automated and Explainable Algorithm for the Prediction of Malignant Transformation in Oral Epithelial Dysplasia

This repository provides the inference code for the models used for predicting slide-level malignancy transformation in OED. Link to preprint. <br />

Currently these scripts require the user to have already generated patch-level (nuclear) features for use within this model. However, we will add these scripts in the near future.

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
