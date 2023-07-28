import os
from torch.multiprocessing import Pool, RLock, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import cv2
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from tiatoolbox.utils.misc import imread, imwrite
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader, WSIMeta
from tiatoolbox.tools import patchextraction

from utils.utils import colour2gray, white2binary, mask2epithmask

import sys
sys.path.append('/data/ANTICIPATE/outcome_prediction/MIL/pipeline/code/')
from utils.features.nuclear import get_nuc_features
from models.resnet_custom import resnet50_baseline

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_nuc_dict(nuc_data, bbox, color_dict, layers=None, draw=None):
    # Layers argument to refine the epithelial classes following HoVer-Net+ postprocessing
    start_x, start_y, end_x, end_y = bbox
    count = 0
    new_dict = {}
    for n in nuc_data:
        centroid = np.asarray(nuc_data[n]['centroid'])
        if (start_x <= centroid[0] < end_x) and (start_y <= centroid[1] < end_y):
            bbox = np.asarray(nuc_data[n]['box'])
            cntrs = np.asarray(nuc_data[n]['contour'])
            centroid_new = centroid.copy()
            centroid_new[0] = centroid[0] - start_x
            centroid_new[1] = centroid[1] - start_y
            bbox_new = bbox.copy()
            bbox_new[0] = bbox[0] - start_x
            bbox_new[2] = bbox[2] - start_x
            bbox_new[1] = bbox[1] - start_y
            bbox_new[3] = bbox[3] - start_y
            cntrs_new = cntrs.copy()
            cntrs_new[:,0] = cntrs[:,0] - start_x
            cntrs_new[:,1] = cntrs[:,1] - start_y
            nuc_type = nuc_data[n]['type']
            count += 1
            nuc_dict = {
                'bbox': bbox_new.tolist(),
                'centroid': centroid_new.tolist(),
                'contour': cntrs_new.tolist(),
                'type_prob': nuc_data[n]['prob'],
                'type': nuc_data[n]['type'],
            }
            if layers is not None:
                if nuc_type in [2,3,4]:
                    tissue_type = int(layers[int(centroid_new[1]), int(centroid_new[0])])
                    nuc_dict['type'] = tissue_type
            new_dict[str(count)] = nuc_dict        
            if draw is not None:
                draw = cv2.drawContours(draw, [cntrs_new], -1, color_dict[nuc_type], 2).astype('uint8')
    if draw is not None:
        return new_dict, draw
    else:
        return new_dict

def create_feature_patches(
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
    output_dir=None,
    viz=False,
    ):
    
    case, _ = os.path.basename(wsi_path).split(".")
    if output_dir is not None:
        output_dir_ftrs = os.path.join(output_dir, f'{output_res}-mpp_{patch_size}_{stride}_epith-{epith_thresh}')

    if viz:
        over_dir = os.path.join(output_dir_ftrs, 'overlays')
        os.makedirs(over_dir, exist_ok=True)

    wsi = WSIReader.open(wsi_path)
    meta = WSIMeta(
        slide_dimensions=tuple(wsi.slide_dimensions(layer_res, 'mpp')), level_downsamples=[1, 2, 4, 8],
        mpp=tuple(wsi.convert_resolution_units(layer_res, 'mpp')['mpp']), objective_power=20, axes='YXS'
    )

    try:
        layers = np.load(mask_path)
        layers_new = layers.copy()
        layers_new[layers_new == 1] = 0
        layers_new[layers_new >= 2] = 1
        layer_mask = VirtualWSIReader(layers_new*255, info=meta)
    except:
        print(f'Failed for case {case}')
        return

    if np.max(layers_new) == 0: # i.e. no epith detected
        return

    if (feature_type == 'nuclear') or (feature_type == 'both'):
        try:
            inst_dict = joblib.load(nuclei_path)
        except:
            print(f'Failed for case {case}, no nuclei dictionary')
            return

    img_patches = patchextraction.get_patch_extractor(
        input_img=wsi,
        input_mask=layer_mask,
        method_name="slidingwindow",
        patch_size=patch_size,
        resolution=output_res,
        units="mpp", #"power",
        stride=stride,
        pad_mode="constant",
        pad_constant_values=255,
        within_bound=False,
    )

    coords = []
    deep_coords = []
    coords_batch = []
    features_all = []
    deep_features_all = []
    batch_size = 32
    patches = []
    count = 0

    if (feature_type == 'resnet') or (feature_type == 'both'):
        model = resnet50_baseline(pretrained=True)
        model = model.to(device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()

    for patch in tqdm(img_patches):
        # 1. Get layer mask
        item = img_patches.n-1
        bounds = img_patches.coordinate_list[item]
        patch_mask = layer_mask.read_bounds(
                    bounds,
                    resolution=output_res,
                    units="mpp",
                    interpolation="nearest",
                    pad_mode="constant",
                    pad_constant_values=0,
                    coord_space="resolution",
        )
        patch_mask_g = colour2gray(patch_mask, colour_dict)
        epith_mask_binary = white2binary(patch_mask)
        patch_name = f'{case}_{bounds[0]}_{bounds[1]}_{bounds[2]}_{bounds[3]}'

        if np.mean(epith_mask_binary) < epith_thresh:
            continue

        if (feature_type == 'nuclear') or (feature_type == 'both'):
        # 2. Check if contains tissue...
        # Then no nuclei present - skip patch
            if len(inst_dict) <= 3:
                continue
            
            # 3. [Optional] Refine nuclei classes
            nuc_dict, overlay = get_nuc_dict(inst_dict, bounds, colour_dict, patch_mask_g, patch)
            try:
                morph_df, spatial_df = get_nuc_features(nuc_dict, patch_name, nr_types=2)
                ftrs_df = pd.concat([morph_df, spatial_df], axis=0)
            except:
                continue

            features_all.append(ftrs_df)
            coords.append(bounds)
            if viz:
                os.makedirs(os.path.join(over_dir, case), exist_ok=True)
                imwrite(os.path.join(over_dir, case, patch_name+'.png'), overlay)

        if (feature_type == 'resnet') or (feature_type == 'both'):
            count += 1
            patches.append(patch)
            coords_batch.append(bounds)
            if (len(patches) % batch_size == 0) or (item == len(img_patches.coordinate_list)-1):
                batch = np.stack(patches, axis=3).transpose(3,2,0,1)
                batch = torch.Tensor(batch)
                coords_batch = np.stack(coords_batch)       
                with torch.no_grad():
                    batch = batch.to(device, non_blocking=True)
                    features = model(batch)
                    features = features.cpu().numpy()
                deep_features_all.extend(features)
                deep_coords.extend(coords_batch)
                patches = []
                coords_batch = []

    if (feature_type == 'resnet') or (feature_type == 'both'):
        if len(patches) != 0:
            batch = np.stack(patches, axis=3).transpose(3,2,0,1)
            batch = torch.Tensor(batch)
            coords_batch = np.stack(coords_batch)       
            with torch.no_grad():
                batch = batch.to(device, non_blocking=True)
                features = model(batch)
                features = features.cpu().numpy()
            deep_features_all.extend(features)
            deep_coords.extend(coords_batch)
            patches = []
            coords_batch = []   

    if feature_type == 'nuclear':
        return features_all, coords
    elif feature_type == 'deep':
        return deep_features_all, deep_coords
    elif feature_type == 'both':
        return features_all, coords, deep_features_all, deep_coords


def create_image_patches(
    wsi_path,
    mask_path,
    layer_col_dict,
    layer_res,
    patch_size,
    stride,
    out_res,
    epith_thresh
    ):
    
    wsi = WSIReader.open(wsi_path)
    mask = imread(mask_path)

    meta = WSIMeta(
        slide_dimensions=tuple(wsi.slide_dimensions(layer_res, 'mpp')), level_downsamples=[1, 2, 4, 8],
        mpp=tuple(wsi.convert_resolution_units(layer_res, 'mpp')['mpp']), objective_power=5, axes='YXS'
    )

    layers_new = mask2epithmask(mask, layer_col_dict, labels=[2,3,4])
    layer_mask = VirtualWSIReader(layers_new, info=meta)

    if np.max(layers_new) == 0: # i.e. no epith detected
        return

    patches = []
    patch_rois = []
    img_patches = patchextraction.get_patch_extractor(
        input_img=wsi,
        input_mask=layer_mask,
        method_name="slidingwindow",
        patch_size=patch_size,
        resolution=out_res,
        units="mpp", #"power",
        stride=stride,
        pad_mode="constant",
        pad_constant_values=255,
        within_bound=False,
    )

    for patch in img_patches:
        # 1. Get layer mask
        item = img_patches.n-1
        bounds = img_patches.coordinate_list[item]
        patch_mask = layer_mask.read_bounds(
                    bounds,
                    resolution=out_res,
                    units="mpp", #"power",
                    interpolation="nearest",
                    pad_mode="constant",
                    pad_constant_values=0,
                    coord_space="resolution",
        )
        epith_mask_binary = white2binary(patch_mask)

        if np.mean(epith_mask_binary) < epith_thresh:
            continue

        patches.append(patch)
        patch_rois.append(bounds)

    return patches, patch_rois
