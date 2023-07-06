import os
import glob
multi_gpu = True
if multi_gpu==False:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import shutil
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from dataloader.mil_reader import featuresdataset_wsi_inference

from tiatoolbox.utils.misc import imread, imwrite
from tiatoolbox.wsicore.wsireader import WSIReader, OpenSlideWSIReader, VirtualWSIReader, WSIMeta
from tiatoolbox.tools import patchextraction
import cv2

from models.net_desc import MLP, FC
from utils.utils import get_heatmap, build_heatmap
from utils.patch_generation import create_morph_patches, create_image_patches

# cnn inference 
def inference(loader, model, batch_size):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset), 2)
    preds = torch.FloatTensor(len(loader.dataset))
    coords_all = torch.FloatTensor(len(loader.dataset), 4)
    with torch.no_grad():
        for i, (inputs, coords) in enumerate(loader):
            inputs = inputs.cuda()
            coords = coords.cuda()
            output = model(inputs)
            y = F.softmax(output, dim=1)
            _, pr = torch.max(output.data, 1)
            preds[i * batch_size:i * batch_size + inputs.size(0)] = pr.detach().clone()
            probs[i * batch_size:i * batch_size + inputs.size(0)] = y.detach().clone()
            coords_all[i * batch_size:i * batch_size + inputs.size(0)] = coords.detach().clone()
    return probs.cpu().numpy(), preds.cpu().numpy(), coords_all.cpu().numpy()

COL_DICT = {
    0: (0,0,0),
    1: (255, 165, 0),
    2: (255, 0, 0),
    3: (0, 255, 0),
    4: (0, 0, 255)
}
LAYER_RESOLUTION = 2
OUTPUT_RESOLUTION = 0.5 #20
PATCH_SIZE = 512
STRIDE = 128 # 256
EPITH_THRESH = 0.5 #0.1
HEATMAP_RES = 2

cohort = 'oed'
method = 'resnet34'
features = 'raw_images'
data_file = f'/data/data/ANTICIPATE/sheffield/new_clinical_data_221222/sheffield_data_c1,3,4_{cohort}_5-folds.csv'
input_wsis = [
    '/data/data/ANTICIPATE/sheffield/all/oed/', # cohort 3
    '/data/data/ANTICIPATE/sheffield/cohort_4/wsis/', # cohort 4
    '/data/data/ANTICIPATE/sheffield/old_oed_wsi/tmp/', # cohort 1
]
input_hovernets = [
    '/data/ANTICIPATE/outcome_prediction/MIL/data/sheffield/segmentations/oed/',
    '/data/ANTICIPATE/outcome_prediction/MIL/data/sheffield_c4/segmentations/',
    '/data/ANTICIPATE/outcome_prediction/MIL/data/sheffield_c1/segmentations/',
]
input_root = '/data/ANTICIPATE/outcome_prediction/MIL/idars/output/cohort_1,3,4/transformation/resnet34/raw_images/oed/'
outcome = 'transformation'
nr_repeats = 3
nr_folds = 5
# wsi_exts = ['ndpi', 'svs']
proc_mpp = 0.5
batch_size = 256
workers = 8

output_folder = os.path.join(input_root, 'heatmaps')
os.makedirs(output_folder, exist_ok=True)

dataset = pd.read_csv(data_file)

if method == 'mlp':
    if features == 'morph_features_156ftrs':
        nr_ftrs = 156
        nr_hidden = 64
    elif features == 'deep_features':
        nr_ftrs = 1024
        nr_hidden = 512
    elif features == 'morph_features_104_64ftrs':
        nr_ftrs = 168
        nr_hidden = 64 #128
    
    if method == 'mlp':
        model = MLP(d=nr_ftrs, hidden_d=nr_hidden, nr_classes=2)
    elif method =='fc':
        model = FC(d=nr_ftrs, nr_classes=2)

elif 'resnet34' in method:
    model = models.resnet34(True) #pretrained resnet34
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

model.cuda()
# Distribute across 2 gpus
if multi_gpu == True:
    model = nn.DataParallel(model, device_ids=[0, 1])
    print('running on multi gpu ...')

cudnn.benchmark = True

summary_path = f'{input_root}/metrics_summary*.csv'
summary_path = glob.glob(summary_path)[0]
best_metric = summary_path.split('summary_')[1].split('.csv')[0]
summary_info = pd.read_csv(summary_path, index_col=[0])

for repeat in [3]:#range(1, nr_repeats+1):
    out_folder = os.path.join(output_folder, f'repeat_{repeat}')
    os.makedirs(out_folder, exist_ok=True)
    for fold in range(nr_folds):      # number of runs for each fold
        torch.cuda.empty_cache()
        input_path =  os.path.join(input_root, f'repeat_{repeat}/best{fold}')     # path to store outputs
        test_data = dataset[dataset[f'{outcome}_fold_{repeat}']==fold][['slide_name', outcome, 'cohort', 'vendor']]
        test_data = test_data.rename(columns={"slide_name": "wsi", outcome: "label"})     

        for r, r_data in test_data.iterrows():
            case = r_data['wsi']
            cohort = r_data['cohort']
            print(f'Repeat {repeat}, fold {fold}: Processing case {case}')
            if cohort == 1:
                wsi_path = glob.glob(os.path.join(input_wsis[2], case + '*'))[0]
                mask_path = input_hovernets[2]
            elif cohort == 3:
                wsi_path = glob.glob(os.path.join(input_wsis[0], case + '*'))[0]
                mask_path = input_hovernets[0]
            elif cohort == 4:
                wsi_path = glob.glob(os.path.join(input_wsis[1], case + '*'))[0]
                mask_path = input_hovernets[1]
                
            # defining data transform
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.1, 0.1, 0.1])
            trans_Valid = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            
            ###### get patches here..... #######
            if features == 'raw_images':
                patches, rois = create_image_patches(
                    wsi_path, mask_path, COL_DICT, LAYER_RESOLUTION, PATCH_SIZE,
                    STRIDE, OUTPUT_RESOLUTION, epith_thresh=EPITH_THRESH
                    )
            else:
                patches, rois = create_morph_patches(
                    'nuclear', wsi_path, mask_path, COL_DICT, LAYER_RESOLUTION, PATCH_SIZE,
                    STRIDE, OUTPUT_RESOLUTION, epith_thresh=EPITH_THRESH
                    )
                trans_Valid=None     
                  
            #loading data
            test_dset = featuresdataset_wsi_inference(
                patches, rois, layer_col_dict=COL_DICT, 
                transform=trans_Valid, raw_images=features=='raw_images')
            print('loaded dataset')
            test_loader = torch.utils.data.DataLoader(
                test_dset,
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=False)

            # loading best checkpoint wrt to auroc on test set
            ch = torch.load(os.path.join(input_path, 'checkpoint_best_AUC.pth'))
            cutoff = summary_info.loc[(summary_info['repeat']==str(repeat)) & (summary_info['fold']==str(fold))]['cutoff'].item()
            state_dict = ch['state_dict']
            # state_dict = convert_state_dict(state_dict)
            model.load_state_dict(state_dict)

            #infernece
            test_probs, test_preds, test_coords = inference(test_loader, model, batch_size)
            test_probs_1 = test_probs[:, 1]
            print('inferred dataset')

            heatmap = build_heatmap(wsi_path, HEATMAP_RES, OUTPUT_RESOLUTION, test_coords, test_probs_1)
            heatmap_col = get_heatmap(heatmap)
            np.save(os.path.join(out_folder, case+'.npy'), heatmap)
            imwrite(os.path.join(out_folder, case+'.png'), heatmap_col)

