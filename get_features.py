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
    ):
    
    case, _ = os.path.basename(wsi_path).split('.')

    layer_mask_path = os.path.join(hovernetplus_dir, 'layers', f'{case}.npy')

    output_dir_ftrs = os.path.join(output_dir, f'{output_res}-mpp_{patch_size}_{stride}_epith-{epith_thresh}')

    over_dir = os.path.join(output_dir_ftrs, 'overlays')
    os.makedirs(over_dir, exist_ok=True)

    wsi = WSIReader.open(wsi_path)
    meta = WSIMeta(
        slide_dimensions=tuple(wsi.slide_dimensions(layer_res, 'mpp')), level_downsamples=[1, 2, 4, 8],
        mpp=tuple(wsi.convert_resolution_units(layer_res, 'mpp')['mpp']), objective_power=20, axes='YXS'
    )

    try:
        layers = np.load(layer_mask_path)
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
            # nuclei_json_path = os.path.join(hovernetplus_dir, group, 'json', f'{case}.json')
            nuclei_path = os.path.join(hovernetplus_dir, 'nuclei', f'{case}.dat')
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

        if np.mean(epith_mask_binary) < epith_thresh:
            continue

        # 2. Check if contains tissue...
        # Then no nuclei present - skip patch
        if len(inst_dict) <= 3:
            continue
    
        # 4. [Optional] Refine nuclei classes
        patch_name = f'{case}_{bounds[0]}_{bounds[1]}_{bounds[2]}_{bounds[3]}'
        nuc_dict, overlay = get_nuc_dict(inst_dict, bounds, colour_dict, patch_mask_g, patch)


        if (feature_type == 'nuclear') or (feature_type == 'both'):

            try:
                morph_df, spatial_df = get_nuc_features(nuc_dict, patch_name, nr_types=2)
                ftrs_df = pd.concat([morph_df, spatial_df], axis=0)
            except:
                continue

            features_all.append(ftrs_df)
            os.makedirs(os.path.join(over_dir, case), exist_ok=True)
            imwrite(os.path.join(over_dir, case, patch_name+'.png'), overlay)
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

