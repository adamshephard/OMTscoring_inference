# OMTscoring: A Fully Automated and Explainable Algorithm for the Prediction of Malignant Transformation in Oral Epithelial Dysplasia

This repository provides the inference code for the models used for predicting slide-level malignancy transformation in OED. Link to preprint [here](https://arxiv.org/abs/2307.03757). <br />

The first step in this pipeline is to use HoVer-Net+ (see original paper [here](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/html/Shephard_Simultaneous_Nuclear_Instance_and_Layer_Segmentation_in_Oral_Epithelial_Dysplasia_ICCVW_2021_paper.html)) to segment the epithelium and nuclei. We have used the TIAtoolbox (see paper [here](https://www.nature.com/articles/s43856-022-00186-5)) implementation of HoVer-Net in the below scripts. Next, we generate patch-level morphological and spatial features to use in our OMTscoring pipeline. After this, we perform the OMTscoring using our pre-trained MLP model.

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

- `run_segmentation.py`: hovernetplus inference script
- `create_features.py`: script to generate features for the final MLP model (using output from above script)
- `h5_bag2tiles.py`: script to get features into the correct format
- `run_infer.py`: main inference script for OMTscoring
- `create_heatmaps.py`: script to generate heatmaps (need tidying up)

## Inference

### Data Format
Input: <br />
- WSIs supported by [OpenSlide](https://openslide.org/), including `svs`, `tif`, `ndpi` and `mrxs`.

### Model Weights

Model weights obtained from training ....

If any of the above checkpoints are used, please ensure to cite the corresponding paper.

### Usage

#### Segmentation with HoVer-Net+

The first stage is to run HoVer-Net+ on the WSIs to generate epithelial and nuclei segmentations. This can be quite slow as run at 0.5mpp. Ensure to change the  `input_wsi_dir` and `output_dir` arguments within the file to ensure they are pointing towards the correct directories.

Usage: <br />
```
  python run_segmentation.py
```

#### Feature Generation

The second stage is to tesselate the image into smaller patches and generate correpsonding patch-level morphological and spatial features using the nuclei/layer segmentations. Ensure to change the  `input_wsi_dir`, `hovernetplus_output_dir`, and `feature_output_dir` arguments within the file to ensure they are pointing towards the correct directories. Note the `hovernetplus_output_dir` is the output directory from the previous step.

Usage: <br />
```
  python create_features.py
```

We then need to adjust the patch output to be in the right format (one file per tile). We can this using the following script. Make sure to change the following arguments: `input_folder` and `output_folder`.

Usage: <br />
```
  python h5_bag2tiles.py
```

#### OMTscoring Inference

The final stage is to infer using the MLP on the tiles (and their features) generated in the previosu steps. Ensure to change the  `checkpoint_path`, `data_file`, `data_path` and `output` arguments within the file to ensure they are pointing towards the correct directories/files.

Usage: <br />
```
  python run_infer.py
```

#### OMTscore Heatmaps

We can also generate heatmaps for these images. Ensure to change the  `input_checkpoint_path`, `input_wsi_dir`, `hovernetplus_output_dir` and `heatmap_output_dir` arguments within the file to ensure they are pointing towards the correct directories/files. Also change the `stride` from 128 create smoother images. However, a decreased stride by 2X wll increase the processing time by 2X.
    
Usage: <br />
```
  python create_heatmaps.py
```

