"""
Inference scripts for OMTscoring. This generates average results over multiple repeats
of cross-validation. E.g. For internal testing.
Inference is based on the IDaRS inference pipeline.
"""
import os
import glob
multi_gpu = True
if multi_gpu==False:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import auc, roc_curve, f1_score, precision_recall_curve, average_precision_score

from dataloader.mil_reader import featuresdataset_inference
from models.net_desc import MLP, FC
from utils.metrics import compute_aggregated_predictions, compute_aggregated_probabilities, group_avg_df, get_topk_patches, get_bottomk_patches

### Inputs Files and Paramaters ###
problem = 'oed'                         # prediction problem.
folds = 5                               # number of folds
repeats = 3                             # number of repeats of each fold i.e. repeated cross-validation
batch_size = 256          
nepochs = 100                           # number of epochs
workers = 6                             # number of data loading workers
test_every = 1                          # test on val every (default: 1
r = 45                                  # how many random tiles to consider
k = 5                                   # how many top k tiles to consider
loss = 'SCE'                            # two options default = cross entropy ('CE') or symmetric cross entropy ('SCE')
method = 'mlp'                          # model being used, for paper use MLP, but alternatives are FC and ResNet34
features = 'morph_features_104_64ftrs'  # input features. Could also be deep features e.g. resnet
outcome = 'transformation'              # prediction outcome
# CSV file containing fold information and targets per slide
data_file = f'/data/data/ANTICIPATE/sheffield/new_clinical_data_221222/sheffield_data_c1,3,4_oed_5-folds.csv'  
# Location of features:
data_path = [
    '/data/ANTICIPATE/outcome_prediction/MIL/data/sheffield/patch_features_4/0.5-mpp_512_256_epith-0.5_156_122ftrs/nuclear/tiles_pt_files/',
    '/data/ANTICIPATE/outcome_prediction/MIL/data/sheffield_c4/patch_features_4/0.5-mpp_512_256_epith-0.5_156_122ftrs/nuclear/tiles_pt_files/',
    '/data/ANTICIPATE/outcome_prediction/MIL/data/sheffield_c1/patch_features_4/0.5-mpp_512_256_epith-0.5_156_122ftrs/nuclear/tiles_pt_files/',
]
# Output folder
output = f'/data/ANTICIPATE/outcome_prediction/MIL/idars/output/cohort_1,3,4/{outcome}/{method}/{features}_{nepochs}eps/'
# Swap filename for problem when require raw images (png) instead of features (.pt)
swap = ['patch_features_4/0.5-mpp_512_256_epith-0.5_156_122ftrs/nuclear/tiles_pt_files/', 'image_patches/0.5-mpp_512_256_epith-0.5_images/']

# cnn inference 
def inference(loader, model, cutoff):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset), 2)
    preds = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, (inputs, target) in enumerate(loader):
            inputs = inputs.cuda()
            target, wsi_name = target
            target = target.cuda()
            output = model(inputs)
            y = F.softmax(output, dim=1)
            _, pr = torch.max(output.data, 1)
            preds[i * batch_size:i * batch_size + inputs.size(0)] = pr.detach().clone()
            probs[i * batch_size:i * batch_size + inputs.size(0)] = y.detach().clone()
    return probs.cpu().numpy(), preds.cpu().numpy()


### Main Script ###
temp_input = os.path.join(output, problem)
dataset = pd.read_csv(data_file)

summary_path = f'{temp_input}/metrics_summary*.csv'
summary_path = glob.glob(summary_path)[0]
best_metric = summary_path.split('summary_')[1].split('.csv')[0]
summary_info = pd.read_csv(summary_path, index_col=[0])

f1s = []
roc_aucs = []
for repeat in range(1, repeats+1): # number of repeats
    input =  os.path.join(temp_input, f'repeat_{repeat}')
    path_fold = input
    
    for fold in range(folds): # number of folds
        torch.cuda.empty_cache()
        input =  os.path.join(path_fold, f'best{fold}') # path to store outputs

        test_data = dataset[dataset[f'{outcome}_fold_{repeat}']==fold][['slide_name', outcome, 'cohort', 'vendor']]
        test_data = test_data.rename(columns={"slide_name": "wsi", outcome: "label"})     

        if "raw_images" in features: # e.g raw images to train CNN
            model = models.resnet34(True) # pretrained resnet34
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            # defining data transform
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.1, 0.1, 0.1])
            trans_Valid = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            raw_images = True
            
        else: # e.g. MLP/FC
            if features == 'morph_features_156ftrs':
                nr_ftrs = 156
                nr_hidden = 64
            elif features == 'deep_features':
                nr_ftrs = 1024
                nr_hidden = 512
            elif features == 'morph_features_104_64ftrs':
                nr_ftrs = 168
                nr_hidden = 64 #128
            trans_Valid = None
            raw_images = False

            if method == 'mlp':
                model = MLP(d=nr_ftrs, hidden_d=nr_hidden, nr_classes=2)
            elif method =='fc':
                model = FC(d=nr_ftrs, nr_classes=2)

        model.cuda()
        cudnn.benchmark = True

        #loading data
        # train set
        test_dset = featuresdataset_inference(data_path=data_path, data_frame=test_data, transform=trans_Valid, raw_images=True)

        test_loader = torch.utils.data.DataLoader(
            test_dset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=False)

        ## test set inference
        # loading best checkpoint wrt to auroc on test set
        ch = torch.load(os.path.join(input, 'checkpoint_best_AUC.pth'))
        cutoff = summary_info.loc[(summary_info['repeat']==str(repeat)) & (summary_info['fold']==str(fold))]['cutoff'].item()
        model.load_state_dict(ch['state_dict'])

        test_dset.setmode(1)
        #infernece
        test_probs, test_preds = inference(test_loader, model, cutoff)
        test_probs_1 = test_probs[:, 1]
        top_prob_test = np.nanmax(test_probs, axis=1)

        ## aggregation of tile scores into slide score
        test_slide_mjvt, test_slide_mjvt_raw  = compute_aggregated_predictions(np.array(test_dset.slideIDX), test_preds)
        test_slide_avg, test_slide_max, test_slide_sum, test_slide_md, test_slide_gm, test_slide_avgtop  = compute_aggregated_probabilities(np.array(test_dset.slideIDX), test_probs_1)
        test_slide_topk_patches = get_topk_patches(np.array(test_dset.slideIDX), np.array(test_dset.tiles), test_probs_1, k)
        test_slide_bottomk_patches = get_bottomk_patches(np.array(test_dset.slideIDX), np.array(test_dset.tiles), test_probs_1, k)
        
        test_slide_avgt5 = group_avg_df(np.array(test_dset.slideIDX), test_probs_1)

        if best_metric == 'med':
            test_pred = test_slide_md
        elif best_metric == 'mj vote':
            test_pred = test_slide_mjvt
        elif best_metric == 'mj vote raw':
            test_pred = test_slide_mjvt_raw
        elif best_metric == 'avg prob':
            test_pred = test_slide_avg
        elif best_metric == 'max prob':
            test_pred = test_slide_max
        elif best_metric == 'sump':
            test_pred = test_slide_sum
        elif best_metric == 'gmp':
            test_pred = test_slide_gm
        elif best_metric == 'avgtop':
            test_pred = test_slide_avgtop 
        elif best_metric == 'top5':
            test_pred = test_slide_avgt5     

        slides = test_dset.slides
        cohorts = test_dset.cohorts
        slides = [os.path.basename(i) for i in slides]
        y_pred = [1 if i >= cutoff else 0 for i in test_pred]
        y_true = test_dset.targets
        fpr, tpr, thresholds = roc_curve(y_true, test_pred)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_true, y_pred)
        f1s.append(f1)
        roc_aucs.append(roc_auc)
        precision, recall, _ = precision_recall_curve(y_true, test_pred)
        average_precision = average_precision_score(y_true, test_pred)
        data_out = pd.DataFrame({'case':slides, 'y_true': y_true, 'y_pred': y_pred, 'cohort': cohorts})
        out_name = os.path.join(input, 'predictions.csv')
        data_out.to_csv(out_name, index=False)

        # save top patches
        for s, slide in enumerate(slides):
            out_patch_dir = os.path.join(output, problem, f'repeat_{repeat}/best{fold}', f'top_{k}_patches', f'cohort_{cohorts[s]}', slide)
            os.makedirs(out_patch_dir, exist_ok=True)
            for patch in test_slide_topk_patches[s]:
                shutil.copy(patch, out_patch_dir)
                if "raw_images" not in features:
                    img_patch = patch.replace(swap[0], swap[1])
                    img_patch = img_patch.replace('.pt', '.png')
                shutil.copy(img_patch, out_patch_dir)
                
        # save bottom patches
        for s, slide in enumerate(slides):
            out_patch_dir = os.path.join(output, problem, f'repeat_{repeat}/best{fold}', f'bottom_{k}_patches', f'cohort_{cohorts[s]}', slide)
            os.makedirs(out_patch_dir, exist_ok=True)
            for patch in test_slide_bottomk_patches[s]:
                shutil.copy(patch, out_patch_dir)
                if "raw_images" not in features:
                    img_patch = patch.replace(swap[0], swap[1])
                    img_patch = img_patch.replace('.pt', '.png')
                    shutil.copy(img_patch, out_patch_dir)   

        m_data = pd.DataFrame({'case':slides, 'y_true': y_true, f'y_pred_repeat{repeat}': y_pred, 'cohort': cohorts})
        
        if fold == 0:
            m_repeat_data = m_data
        else:
            m_repeat_data = pd.concat([m_repeat_data, m_data])

        if (fold == 4) & (repeat == 1):
            merged_data = m_repeat_data
        elif (fold == 4) & (repeat != 1):
            merged_data = merged_data.merge(m_repeat_data, 'inner', on=['case', 'y_true', 'cohort'])

# Now combine results....
out_name2 = os.path.join(output, problem, f'summary_predictions.csv')
merged_data.to_csv(out_name2, index=False)

summary_df = pd.DataFrame({'repeat':[r for r in range(1,repeats+1) for f in range(folds)], 'fold': [f for r in range(repeats) for f in range(folds)], 'f1_score': f1s, 'auroc': roc_aucs})
df_mean = summary_df.mean()
df_std = summary_df.std()
summary_df.loc['mean'] = df_mean
summary_df.loc['std'] = df_std
summary_df['repeat']['mean'] = "-"
summary_df['repeat']['std'] = "-"
summary_df['fold']['mean'] = "-"
summary_out_name = os.path.join(output, problem, f'summary_metrics.csv')
summary_df.to_csv(summary_out_name)
