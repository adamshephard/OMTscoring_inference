"""
Create heatmaps for OMTscoring model across the WSI.

Usage:
  create_heatmaps.py [options] [--help] [<args>...]
  create_heatmaps.py --version
  create_heatmaps.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.

  --input_dir=<string>        Path to input directory containing slides or images.
  --hovernetplus_dir=<string> Path to hovernetplus directory.
  --checkpoint_path=<string>  Path to MLP checkpoint.
  --output_dir=<string>       Path to output directory to save results.
  --batch_size=<n>            Batch size. [default: 256]

Use `create_heatmaps.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from dataloader.mil_reader import featuresdataset_wsi_inference

from tiatoolbox.utils.misc import imwrite

from models.net_desc import MLP, FC
from utils.utils import get_heatmap, build_heatmap
from utils.patch_generation import create_feature_patches, create_image_patches

from tqdm import tqdm

# cnn inference 
def inference(loader, model, batch_size):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset), 2)
    preds = torch.FloatTensor(len(loader.dataset))
    coords_all = torch.FloatTensor(len(loader.dataset), 4)
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, (inputs, coords) in tqdm(enumerate(loader)):
                inputs = inputs.cuda()
                coords = coords.cuda()
                output = model(inputs)
                y = F.softmax(output, dim=1)
                _, pr = torch.max(output.data, 1)
                preds[i * batch_size:i * batch_size + inputs.size(0)] = pr.detach().clone()
                probs[i * batch_size:i * batch_size + inputs.size(0)] = y.detach().clone()
                coords_all[i * batch_size:i * batch_size + inputs.size(0)] = coords.detach().clone()
                pbar.update(1)
    return probs.cpu().numpy(), preds.cpu().numpy(), coords_all.cpu().numpy()


def process(
    model,
    features,
    wsi_path,
    input_checkpoint_path,
    hovernetplus_dir,
    output_dir,
    colour_dict,
    patch_size,
    stride,
    proc_res,
    output_res,
    layer_res,
    epith_thresh,
    batch_size
    ):
    
    case = os.path.basename(wsi_path).split(".")[0]
    mask_path = os.path.join(hovernetplus_dir, "layers", f"{case}.npy")
    nuclei_path = os.path.join(hovernetplus_dir, "nuclei", f"{case}.dat")
            
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
            wsi_path, mask_path, colour_dict, layer_res, patch_size,
            stride, proc_res, epith_thresh
            )
    else:
        patches, rois = create_feature_patches(
            'nuclear', wsi_path, mask_path, nuclei_path, colour_dict, patch_size,
            stride, proc_res, layer_res, epith_thresh, output_dir=None, viz=False
            )
        trans_Valid=None
        for idx, ftr_df in enumerate(patches):
            tmp = ftr_df.T
            if idx == 0:
                ftr_df_all = tmp
            else:
                ftr_df_all = pd.concat([ftr_df_all, tmp], axis=0)
            ftr_df_all.reset_index(drop=True, inplace=True)
        patches = ftr_df_all.iloc[:, 0:].to_numpy()
        rois = np.vstack(rois)
        
    # Load data
    test_dset = featuresdataset_wsi_inference(
        patches, rois, layer_col_dict=colour_dict, 
        transform=trans_Valid, raw_images=features=='raw_images')
    
    print('loaded dataset')
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=False)

    # loading best checkpoint wrt to auroc on test set
    ch = torch.load(input_checkpoint_path)
    state_dict = ch['state_dict']
    # state_dict = convert_state_dict(state_dict)
    model.load_state_dict(state_dict)

     #run inference
    test_probs, _, test_coords = inference(test_loader, model, batch_size)
    test_probs_1 = test_probs[:, 1]
    print('inferred dataset')

    heatmap = build_heatmap(wsi_path, output_res, proc_res, test_coords, test_probs_1)
    heatmap_col = get_heatmap(heatmap)
    np.save(os.path.join(output_dir, f"{case}.npy"), heatmap)
    imwrite(os.path.join(output_dir, f"{case}.png"), heatmap_col)
    return


if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True)

    if args['--help']:
        print(__doc__)
        exit()

    if args['--input_dir']:
        input_wsi_dir = args['--input_dir']
    else:      
        input_wsi_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/wsis/"

    if args['--hovernetplus_dir']:
        hovernetplus_dir = args['--hovernetplus_dir']
    else:
        hovernetplus_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/hovernetplus/"#  
    
    if args['--checkpoint_path']:
        input_checkpoint_path = args['--checkpoint_path']
    else:
        input_checkpoint_path = "/data/ANTICIPATE/outcome_prediction/MIL/idars/output/train-sheffield_test-belfast,birmingham,brazil/transformation/mlp/morph_features_104_64ftrs_50eps_corrected_belfast_train_thr/oed/repeat_2/best0/checkpoint_best_AUC.pth"
    
    if args['--output_dir']:
        heatmap_output_dir = args['--output_dir']
    else:
        heatmap_output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/heatmaps/"
    
    if args['--batch_size']:
        batch_size = int(args['--batch_size'])
    
    ### Input/Output Parameters ###
    method = "mlp" # alternatives are "resnet34", "resnet34_SA", "resnet34_DA", "resnet34_SA_DA"
    features = 'morph_features_104_64ftrs' # alternatives are "raw_images"
    colour_dict = {
            0: (0,0,0),
            1: (255,165,0),
            2: (255,0,0),
            3: (0,255,0),
            4: (0,0,255)
        }
    patch_size = 512 # desired output patch size
    stride = 128 # stride for sliding window of patches. Decrease stride to make smoother heatmaps (but 0.5x stride = 2x comp. time)
    proc_res = 0.5 # resolution of intermediate patches 
    output_res = 2 # desired resolution of output heatmaps
    layer_res = 0.5 # resolution of layer segmentation from HoVer-Net+ in MPP
    epith_thresh = 0.5 # threshold for ratio of epithelium required to be in a patch to use patch
    
    ### Main ###
    wsi_file_list = glob.glob(input_wsi_dir + "*")
    os.makedirs(heatmap_output_dir, exist_ok=True)
    torch.cuda.empty_cache()

    # Load Model
    if method == 'mlp':
        nr_ftrs = 168
        nr_hidden = 64  

    if method == 'mlp':
        model = MLP(d=nr_ftrs, hidden_d=nr_hidden, nr_classes=2)
    elif method =='fc':
        model = FC(d=nr_ftrs, nr_classes=2)

    elif 'resnet34' in method:
        model = models.resnet34(True) # pretrained resnet34
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
    model.cuda()
    cudnn.benchmark = True
    for wsi_path in wsi_file_list:
        process(
                model,
                features,
                wsi_path,
                input_checkpoint_path,
                hovernetplus_dir,
                heatmap_output_dir,
                colour_dict,
                patch_size,
                stride,
                proc_res,
                output_res,
                layer_res,
                epith_thresh,
                batch_size,
                )
