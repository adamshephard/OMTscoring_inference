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
from tqdm import tqdm

from dataloader.mil_reader import featuresdataset_inference
from models.net_desc import MLP, FC
from utils.metrics import compute_aggregated_predictions, compute_aggregated_probabilities, group_avg_df, get_topk_patches, get_bottomk_patches

### Inputs Files and Paramaters ###
batch_size = 256          
workers = 6                             # number of data loading workers
aggregation_method = "avgtop"           # method for aggregating predictions
cutoff = 0.5                            # cutoff for aggregation method used (found through training)
method = "mlp"                          # model being used, for paper use MLP, but alternatives are FC and ResNet34
features = "morph_features_104_64ftrs"  # input features. Could also be deep features e.g. resnet
outcome = "transformation"              # prediction outcome

# Checkpoint file of model to load
checkpoint_path = ""

# CSV file containing fold information and targets per slide
data_file = ""
  
# Location of features. Stored as indiviudal .tar files containing each tile's features
data_path = ""

# Output folder
output = ""

# cnn inference 
def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset), 2)
    preds = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, (inputs, target) in enumerate(loader):
                inputs = inputs.cuda()
                target, wsi_name = target
                target = target.cuda()
                output = model(inputs)
                y = F.softmax(output, dim=1)
                _, pr = torch.max(output.data, 1)
                preds[i * batch_size:i * batch_size + inputs.size(0)] = pr.detach().clone()
                probs[i * batch_size:i * batch_size + inputs.size(0)] = y.detach().clone()
                pbar.update(1)
    return probs.cpu().numpy(), preds.cpu().numpy()


### Main Script ###
dataset = pd.read_csv(data_file)
    
torch.cuda.empty_cache()

test_data = dataset[dataset['test']==1][['slide_name', outcome, 'cohort', 'vendor']]
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
ch = torch.load(checkpoint_path)
model.load_state_dict(ch['state_dict'])

#inference
test_probs, test_preds = inference(test_loader, model, cutoff)
test_probs_1 = test_probs[:, 1]
top_prob_test = np.nanmax(test_probs, axis=1)

## aggregation of tile scores into slide score
test_slide_mjvt, test_slide_mjvt_raw  = compute_aggregated_predictions(np.array(test_dset.slideIDX), test_preds)
test_slide_avg, test_slide_max, test_slide_sum, test_slide_md, test_slide_gm, test_slide_avgtop  = compute_aggregated_probabilities(np.array(test_dset.slideIDX), test_probs_1)
test_slide_topk_patches = get_topk_patches(np.array(test_dset.slideIDX), np.array(test_dset.tiles), test_probs_1, k)
test_slide_bottomk_patches = get_bottomk_patches(np.array(test_dset.slideIDX), np.array(test_dset.tiles), test_probs_1, k)

test_slide_avgt5 = group_avg_df(np.array(test_dset.slideIDX), test_probs_1)

if aggregation_method == 'med':
    test_pred = test_slide_md
elif aggregation_method == 'mj vote':
    test_pred = test_slide_mjvt
elif aggregation_method == 'mj vote raw':
    test_pred = test_slide_mjvt_raw
elif aggregation_method == 'avg prob':
    test_pred = test_slide_avg
elif aggregation_method == 'max prob':
    test_pred = test_slide_max
elif aggregation_method == 'sump':
    test_pred = test_slide_sum
elif aggregation_method == 'gmp':
    test_pred = test_slide_gm
elif aggregation_method == 'avgtop':
    test_pred = test_slide_avgtop 
elif aggregation_method == 'top5':
    test_pred = test_slide_avgt5     

slides = test_dset.slides
cohorts = test_dset.cohorts
slides = [os.path.basename(i) for i in slides]
y_pred = [1 if i >= cutoff else 0 for i in test_pred]
y_true = test_dset.targets
fpr, tpr, thresholds = roc_curve(y_true, test_pred)
roc_auc = auc(fpr, tpr)
f1 = f1_score(y_true, y_pred)
precision, recall, _ = precision_recall_curve(y_true, test_pred)
average_precision = average_precision_score(y_true, test_pred)
data_out = pd.DataFrame({'case':slides, 'y_true': y_true, 'y_pred': y_pred, 'cohort': cohorts})
out_name = os.path.join(input, 'predictions.csv')
data_out.to_csv(out_name, index=False)

m_data = pd.DataFrame({'case':slides, 'y_true': y_true, 'y_pred': y_pred, 'cohort': cohorts})

# Now combine results....
out_name2 = os.path.join(output, f'predictions.csv')
m_data.to_csv(out_name2, index=False)

summary_df = pd.DataFrame({'f1_score': f1, 'auroc': roc_auc, 'precision': precision, 'recall': recall, 'average_precision': average_precision})
summary_out_name = os.path.join(output, f'summary_metrics.csv')
summary_df.to_csv(summary_out_name)
