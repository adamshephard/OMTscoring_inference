import os
import glob
import h5py
import torch
from tqdm import tqdm

input_folder = '/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/features/0.5-mpp_512_256_epith-0.5/nuclear/h5_files/'
output_folder = '/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/features/0.5-mpp_512_256_epith-0.5/nuclear/tiles_pt_files/'

for wsi_path in tqdm(sorted(glob.glob(input_folder + "/*.h5"))):
    case = os.path.basename(wsi_path).split('.')[0]
    wsi_ftr_dict = h5py.File(wsi_path, 'r')
    for idx, coords in enumerate(wsi_ftr_dict['coords']):
        tile_name = f'{case}_{coords[0]}_{coords[1]}_{coords[2]}_{coords[3]}'
        ftrs = wsi_ftr_dict['features'][idx]
        output_dir = os.path.join(output_folder, case)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(ftrs, os.path.join(output_dir, f'{tile_name}.pt'))