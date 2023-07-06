import os
import glob
from torch.multiprocessing import Pool, RLock, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import json
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from tiatoolbox.utils.misc import imread, imwrite
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader, VirtualWSIReader, WSIMeta
from tiatoolbox.tools import patchextraction

from utils.utils import colour2gray, white2binary, mask2epithmask

import sys
sys.path.append('/data/ANTICIPATE/outcome_prediction/MIL/pipeline/code/')
from utils.features.nuclear import get_nuc_features
from models.resnet_custom import resnet50_baseline

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_nuc_dict(jsdata, bbox, output_res, color_dict, layers=None, draw=None):
    # Layers argument to refine the epithelial classes following HoVer-Net+ postprocessing
    start_x, start_y, end_x, end_y = bbox
    count = 0
    new_dict = {}
    for n in jsdata['nuc']:
        centroid = np.asarray(jsdata['nuc'][n]['centroid'])
        if (start_x <= centroid[0] < end_x) and (start_y <= centroid[1] < end_y):
            bbox = np.asarray(jsdata['nuc'][n]['bbox'])
            cntrs = np.asarray(jsdata['nuc'][n]['contour'])
            centroid_new = centroid.copy()
            centroid_new[0] = centroid[0] - start_x
            centroid_new[1] = centroid[1] - start_y
            bbox_new = bbox.copy()
            bbox_new[:,0] = bbox[:,0] - start_x
            bbox_new[:,1] = bbox[:,1] - start_y
            cntrs_new = cntrs.copy()
            cntrs_new[:,0] = cntrs[:,0] - start_x
            cntrs_new[:,1] = cntrs[:,1] - start_y
            nuc_type = jsdata['nuc'][n]['type']
            count += 1
            nuc_dict = {
                'bbox': bbox_new.tolist(),
                'centroid': centroid_new.tolist(),
                'contour': cntrs_new.tolist(),
                'type_prob': jsdata['nuc'][n]['type_prob'],
                'type': jsdata['nuc'][n]['type'],
            }
            if layers is not None:
                if nuc_type in [2,3,4]:
                    tissue_type = int(layers[int(centroid_new[1]), int(centroid_new[0])])
                    nuc_dict['type'] = tissue_type
            new_dict[str(count)] = nuc_dict        
            if draw is not None:
                draw = cv2.drawContours(draw, [cntrs_new], -1, color_dict[nuc_type], 2).astype('uint8')
    new_dict = {
        # 'mag': OUTPUT_RESOLUTION,
        'mpp': output_res,
        'nuc': new_dict
    }    
    if draw is not None:
        return new_dict, draw
    else:
        return new_dict

def create_morph_patches(feature_type, wsi_path, hovernetplus_dir, layer_col_dict, layer_res, patch_size, stride, output_res, pad_value=255, epith_thresh=0.5):
    case, _ = os.path.basename(wsi_path).split('.')
    layer_mask_path = os.path.join(hovernetplus_dir, 'layers', f'{case}.png')

    wsi = OpenSlideWSIReader(wsi_path)
    meta = WSIMeta(
        slide_dimensions=tuple(wsi.slide_dimensions(layer_res, 'mpp')), level_downsamples=[1, 2, 4, 8],
        mpp=tuple(wsi.convert_resolution_units(layer_res, 'mpp')['mpp']), objective_power=5, axes='YXS'
    )

    try:
        layers = imread(layer_mask_path)
        layers_new = mask2epithmask(layers, layer_col_dict, labels=[2,3,4]) #[2,3]) #,3,4]) # 2 is basal, 3 epith, 4 keratin. Give as list of labels. 
        layer_mask = VirtualWSIReader(layers_new, info=meta)
    except:
        print(f'Failed for case {case}')
        return

    # if any(item in [2,3,4] for item in np.unique(layers)) == False:
    if np.max(layers_new) == 0: # i.e. no epith detected
        return

    if (feature_type == 'nuclear') or (feature_type == 'both'):
        try:
            # nuclei_json_path = os.path.join(hovernetplus_dir, group, 'json', f'{case}.json')
            nuclei_json_path = os.path.join(hovernetplus_dir, 'json', f'{case}.json')
            with open(nuclei_json_path) as f:
                jsdata = json.load(f)
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
        pad_constant_values=pad_value,
        within_bound=False,
    )

    coords = []
    deep_coords = []
    coords_batch = []
    features_all = []
    deep_features_all = []
    features_all_graph = []
    features_all_morph_graph = []
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
                    units="mpp", #"power",
                    interpolation="nearest",
                    pad_mode="constant",
                    pad_constant_values=0,
                    coord_space="resolution",
        )
        patch_mask_g = colour2gray(patch_mask, layer_col_dict)
        # REMEMBER EPITH NUCLEI ARE BEING DEFAULTED TO NOT BE IN KERATIN/BASAL LAYER
        # patch_mask_binary = np.where(patch_mask_g > 0, 1, 0)
        # epith_mask_binary = np.where(patch_mask_g > 1, 1, 0)
        epith_mask_binary = white2binary(patch_mask)

        if np.mean(epith_mask_binary) < epith_thresh:
            continue

        # 2. Check if contains tissue...
        # Then no nuclei present - skip patch
        if not jsdata['nuc']:
            continue
    
        # 4. [Optional] Refine nuclei classes
        patch_name = f'{case}_{bounds[0]}_{bounds[1]}_{bounds[2]}_{bounds[3]}'
        nuc_dict, overlay = get_nuc_dict(jsdata, bounds, output_res, layer_col_dict, patch_mask_g, patch)

        if len(nuc_dict['nuc']) <= 3:
            continue

        if (feature_type == 'nuclear') or (feature_type == 'both'):

            try:
                morph_df, graph_df, morph_df_1 = get_nuc_features(nuc_dict, patch_name, nr_types=2, consolidate=True, reduced_set=True)
            except:
                continue

            features_all.append(np.asarray(morph_df[0].tolist()))
            features_all_morph_graph.append(np.asarray([*(morph_df[0].tolist()), *(graph_df[0].tolist())]))
            features_all_graph.append(np.asarray(graph_df[0].tolist()))

            # os.makedirs(os.path.join(over_dir, case), exist_ok=True)
            # imwrite(os.path.join(over_dir, case, patch_name+'.png'), overlay)
            coords.append(bounds)

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
        return np.stack(features_all_morph_graph), np.stack(coords)
    elif feature_type == 'deep':
        return np.stack(deep_features_all), np.stack(deep_coords)