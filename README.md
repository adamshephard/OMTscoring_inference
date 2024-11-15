# OMTscoring: A Fully Automated and Explainable Algorithm for the Prediction of Malignant Transformation in Oral Epithelial Dysplasia

This repository provides the inference code for the models used for predicting slide-level malignancy transformation in OED. Link to npj Precision Oncology paper [here](https://www.nature.com/articles/s41698-024-00624-8). <br />

The first step in this pipeline is to use HoVer-Net+ (see original paper [here](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/html/Shephard_Simultaneous_Nuclear_Instance_and_Layer_Segmentation_in_Oral_Epithelial_Dysplasia_ICCVW_2021_paper.html)) to segment the epithelium and nuclei. We have used the TIAToolbox (see paper [here](https://www.nature.com/articles/s43856-022-00186-5)) implementation of HoVer-Net+ in the below scripts. Next, we generate patch-level morphological and spatial features to use in our OMTscoring pipeline. After this, we perform the OMTscoring using our pre-trained MLP model.

## Set Up Environment

We use Python 3.10 with the [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) package installed. By default this uses PyTorch 2.0.

```
conda create -n tiatoolbox python=3.10
conda activate tiatoolbox
pip install tiatoolbox
pip install h5py
pip install docopt
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
- `run_omt_scoring.py`: main inference script for OMTscoring
- `create_heatmaps.py`: script to generate heatmaps (need tidying up)

## Inference

### Data Format
Input: <br />
- WSIs supported by [TIAToolbox](https://tia-toolbox.readthedocs.io/en/latest/?badge=latest), including `svs`, `tif`, `ndpi` and `mrxs`.

### Model Weights

The MLP model weights obtained from training on the Sheffield OED dataset: [OED MLP checkpoint](https://drive.google.com/file/d/1yYWO1EAbXgv7eW98c7CxXydW9SdUtMQW/view?usp=sharing). If the model/checkpoint is used, please ensure to cite the corresponding paper.

### Usage

#### Segmentation with HoVer-Net+

The first stage is to run HoVer-Net+ on the WSIs to generate epithelial and nuclei segmentations. This can be quite slow as run at 0.5mpp.

Usage: <br />
```
  python run_segmentation.py --input_dir="/path/to/input/slides/or/images/dir/" --output_dir="/path/to/output/dir/"
```

#### Feature Generation

The second stage is to tesselate the image into smaller patches and generate correpsonding patch-level morphological and spatial features using the nuclei/layer segmentations. Note the `hovernetplus_dir` is the output directory from the previous step.

Usage: <br />
```
  python create_features.py --input_dir="/path/to/input/slides/or/images/dir/" --hovernetplus_dir="/path/to/hovernetplus/output/" --output_dir="/path/to/output/feature/dir/"
```

We then need to adjust the patch output to be in the right format (one file per tile). We can this using the following script. Here, the input directory is the bag-level nuclear features created by the previous line, e.g. `features/0.5-mpp_512_256_epith-0.5/nuclear/h5_files/`.

Usage: <br />
```
  python h5_bag2tiles.py --input_dir="/path/to/input/bag/features/" --output_dir="/path/to/output/tile/features/"
```

#### OMTscoring Inference

The final stage is to infer using the MLP on the tiles (and their features) generated in the previous steps. Here, the `input_ftrs_dir` is the directroy containnig the features created in the previous steps. The `model_checkpoint` path is tot he weights provided above, and the `input_data_file` is the path to the data file describing the slides to process. An example file is provided in `data_file_template.csv`.

Usage: <br />
```
  python run_omt_scoring.py --input_data_file="/path/to/input/data/file/" --input_ftrs_dir="/path/to/input/tile/ftrs/" --model_checkpoint="/path/to/model/checkpoint/" --output_dir="/path/to/output/dir/"
```

#### OMTscore Heatmaps

We can also generate heatmaps for these images. Change the `stride` within the file from 128 to create smoother images. However, a decreased stride by 2X will increase the processing time by 2X.
    
Usage: <br />
```
  python create_heatmaps.py --input_dir="/path/to/input/slides/or/images/dir/" --hovernetplus_dir="/path/to/hovernetplus/output/" --checkpoint_path="/path/to/checkpoint/" --output_dir="/path/to/heatmap/output/dir/"
```


## License

The source code as hosted on GitHub is released under the BSD-3-Clause license. The full text of the licence is included in [LICENSE](https://github.com/adamshephard/OMTscoring_inference/blob/main/LICENSE) file for further details.

Model weights are licensed under [Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider the implications of using the weights under this license. 

## Cite this repository

If you find ODYN useful or use it in your research, please cite our paper:

```
@article{shephard2024,
  title={A fully automated and explainable algorithm for predicting malignant transformation in oral epithelial dysplasia},
  author={Shephard, Adam J and Bashir, Raja Muhammad Saad and Mahmood, Hanya and Jahanifar, Mostafa and Minhas, Fayyaz and Raza, Shan E Ahmed and McCombe, Kris D and Craig, Stephanie G and James, Jacqueline and Brooks, Jill and others},
  journal={npj Precision Oncology},
  volume={8},
  number={1},
  pages={137},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
