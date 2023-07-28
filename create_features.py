"""
Generate morphological/spatial features for MLP (based on HoVer-Net+ output) for OMTscoring.
"""

import os
import glob
from torch.multiprocessing import Pool, RLock, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

from tqdm import tqdm
import numpy as np
import pandas as pd
import joblib
import cv2
import torch
import torch.nn as nn


from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader, WSIMeta
from tiatoolbox.utils.misc import imread, imwrite
from tiatoolbox.tools import patchextraction

from models.resnet_custom import resnet50_baseline
from utils.utils import save_hdf5, colour2gray, white2binary
from utils.features.nuclear import get_nuc_features
from utils.patch_generation import create_feature_patches

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def process(
    feature_type,
    wsi_path,
    hovernetplus_dir,
    output_dir,
    colour_dict,
    patch_size,
    stride,
    output_res,
    layer_res,
    epith_thresh,
    viz=True,
    ):
    
    case, _ = os.path.basename(wsi_path).split('.')
    mask_path = os.path.join(hovernetplus_dir, 'layers', f'{case}.npy')
    nuclei_path = os.path.join(hovernetplus_dir, 'nuclei', f'{case}.dat')
    output_dir_ftrs = os.path.join(output_dir, f'{output_res}-mpp_{patch_size}_{stride}_epith-{epith_thresh}')

    feature_info = create_feature_patches(
        feature_type,
        wsi_path,
        mask_path,
        nuclei_path,
        colour_dict,
        patch_size,
        stride,
        output_res,
        layer_res,
        epith_thresh,
        output_dir,
        viz=viz,
    )

    if feature_type == "nuclear":
        features_all, coords = feature_info
    elif feature_type == 'deep':
        deep_features_all, deep_coords = feature_info
    elif feature_type == 'both':
        features_all, coords, deep_features_all, deep_coords = feature_info

    if (feature_type == 'nuclear') or (feature_type == 'both'):
        for idx, ftr_df in enumerate(features_all):
            tmp = ftr_df.T
            tmp.insert(loc=0, column='coords', value=f"{coords[idx][0]}_{coords[idx][1]}_{coords[idx][2]}_{coords[idx][3]}")
            if idx == 0:
                ftr_df_all = tmp
            else:
                ftr_df_all = pd.concat([ftr_df_all, tmp], axis=0)
            ftr_df_all.reset_index(drop=True, inplace=True)
        features_all = ftr_df_all.iloc[:, 1:].to_numpy()
        asset_dict = {'features': features_all, 'coords': np.stack(coords)}
        h5_dir = os.path.join(output_dir_ftrs, 'nuclear', 'h5_files')
        pt_dir = os.path.join(output_dir_ftrs, 'nuclear', 'pt_files')
        csv_dir = os.path.join(output_dir_ftrs, 'nuclear', 'csv_files')
        os.makedirs(h5_dir, exist_ok=True)
        os.makedirs(pt_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        save_hdf5(os.path.join(h5_dir, f'{case}.h5'), asset_dict, attr_dict= None, mode='w')
        features = torch.from_numpy(features_all)
        torch.save(features, os.path.join(pt_dir, f'{case}.pt'))
        ftr_df_all.to_csv(os.path.join(csv_dir, f'{case}.csv')) 

    if (feature_type == 'resnet') or (feature_type == 'both'):
        asset_dict = {'features': np.stack(deep_features_all), 'coords': np.stack(deep_coords)}
        h5_dir = os.path.join(output_dir_ftrs, 'resnet', 'h5_files')
        pt_dir = os.path.join(output_dir_ftrs, 'resnet', 'pt_files')
        csv_dir = os.path.join(output_dir_ftrs, 'resnet', 'csv_files')
        os.makedirs(h5_dir, exist_ok=True)
        os.makedirs(pt_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        save_hdf5(os.path.join(h5_dir, f'{case}.h5'), asset_dict, attr_dict= None, mode='w')
        features = torch.from_numpy(np.stack(deep_features_all))
        torch.save(features, os.path.join(pt_dir, f'{case}.pt'))
        deep_df = pd.DataFrame.from_records(deep_features_all)
        deep_coords_names = []
        for coord in deep_coords:
            deep_coords_names.append(f"{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}")
        deep_df.insert(loc=0, column='coords', value=np.stack(deep_coords_names))
        deep_df.to_csv(os.path.join(csv_dir, f'{case}.csv'))
    return


if __name__ == "__main__":
    
    ### Input/Output Files ###
    input_wsi_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/wsis/"
    hovernetplus_output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/hovernetplus/"
    feature_output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/features/"
    
    ### Input/Output Parameters ###
    num_processes = 1
    feature_type = "both" # Choose from: "nuclear", "resnet", "both"
    colour_dict = {
            0: (0,0,0),
            1: (255,165,0),
            2: (255,0,0),
            3: (0,255,0),
            4: (0,0,255)
        }
    patch_size = 512 # desired output patch size
    stride = 256 # stride for sliding window of patches
    output_res = 0.5 # desired resolution of output patches
    layer_res = 0.5 # resolution of layer segmentation from HoVer-Net+ in MPP
    epith_thresh = 0.5 # threshold for ratio of epithelium required to be in a patch to use patch
    
    ### Main ###
    wsi_file_list = glob.glob(input_wsi_dir + "*")

    # Start multi-processing
    argument_list = wsi_file_list
    
    num_jobs = len(argument_list)
    pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)
    pbar_format = "Processing cases... |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    pbarx = tqdm(
        total=len(wsi_file_list), bar_format=pbar_format, ascii=True, position=0
    )
    
    def pbarx_update(*a):
        pbarx.update()

    jobs = [pool.apply_async(
            process,
            args=(
                feature_type,
                n,
                hovernetplus_output_dir,
                feature_output_dir,
                colour_dict,
                patch_size,
                stride,
                output_res,
                layer_res,
                epith_thresh,),
            callback=pbarx_update) for n in argument_list]
    
    pool.close()
    result_list = [job.get() for job in jobs]

    pbarx.close()

