"""
Inference script for OMTscoring.

Usage:
  run_omt_scoring.py [options] [--help] [<args>...]
  run_omt_scoring.py --version
  run_omt_scoring.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.

  --input_data_file=<string>  Path to csv file containing fold information and targets per slide.
  --input_ftrs_dir=<string>   Path to folder containing features. Stored as indiviudal .tar files containing each tile's features
  --output_dir=<string>       Path to output directory to save results.
  --model_checkpoint=<string> Path to model checkpoint.
  --mode=<string>             Tile-level or WSI-level mode. [default: wsi]
  --nr_loader_workers=<n>     Number of workers during data loading. [default: 10]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 10]
  --batch_size=<n>            Batch size. [default: 8]

Use `run_omt_scoring.py --help` to show their options and usage
"""


from docopt import docopt
import os
multi_gpu = True
if multi_gpu==False:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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


def process(data_file, data_path, checkpoint_path, output,
        batch_size, workers, aggregation_method, cutoff,
        method, features, outcome, k):
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

    dataset = pd.read_csv(data_file)
        
    torch.cuda.empty_cache()

    test_data = dataset[dataset['test']==1][['slide_name', outcome, 'cohort']]
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
        if features == 'deep_features':
            nr_ftrs = 1024
            nr_hidden = 512
        elif features == 'morph_features_104_64ftrs':
            nr_ftrs = 168
            nr_hidden = 64
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
    test_dset = featuresdataset_inference(data_path=data_path, data_frame=test_data, transform=trans_Valid, raw_images=raw_images)

    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False)

    ## test set inference
    # loading best checkpoint wrt to auroc on test set
    ch = torch.load(checkpoint_path)
    model.load_state_dict(ch['state_dict'])

    #inference
    test_probs, test_preds = inference(test_loader, model)
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
    out_name = os.path.join(output, 'predictions.csv')
    data_out.to_csv(out_name, index=False)

    m_data = pd.DataFrame({'case':slides, 'y_true': y_true, 'y_pred': y_pred, 'cohort': cohorts})

    # Now combine results....
    out_name2 = os.path.join(output, f'predictions.csv')
    m_data.to_csv(out_name2, index=False)

    summary_df = pd.DataFrame({'f1_score': f1, 'auroc': roc_auc, 'precision': precision, 'recall': recall, 'average_precision': average_precision})
    summary_out_name = os.path.join(output, f'summary_metrics.csv')
    summary_df.to_csv(summary_out_name)


if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True)

    if args['--help']:
        print(__doc__)
        exit()

    if args['--input_data_file']:
        input_data_file = args['--input_data_file']
    else:      
        input_data_file = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/inference_data.csv"   
    
    if args['--input_ftrs_dir']:
        input_ftrs_dir = args['--input_ftrs_dir']
    else:
        input_ftrs_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/features/0.5-mpp_512_256_epith-0.5/nuclear/tiles_pt_files/"
        
    if args['--output_dir']:
        output_dir = args['--output_dir']
    else:
        output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/OMTscoring/"
        
    if args['--model_checkpoint']:
        checkpoint_path = args['--model_checkpoint']
    else:
        checkpoint_path = "/data/ANTICIPATE/outcome_prediction/MIL/idars/output/train-sheffield_test-belfast,birmingham,brazil/transformation/mlp/morph_features_104_64ftrs_50eps_corrected_belfast_train_thr/oed/repeat_2/best0/checkpoint_best_AUC.pth"
    

    ### Inputs Files and Paramaters ###
    batch_size = 256          
    workers = 6                             # number of data loading workers
    aggregation_method = "avgtop"           # method for aggregating predictions
    cutoff = 0.0594686280738292             # cutoff for aggregation method used (found through training)
    method = "mlp"                          # model being used, for paper use MLP, but alternatives are FC and ResNet34
    features = "morph_features_104_64ftrs"  # input features. Could also be deep features e.g. resnet
    outcome = "transformation"              # prediction outcome
    k = 5                                   # top/bottom patches to keep
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    process(
        input_data_file, input_ftrs_dir, checkpoint_path, output_dir,
        batch_size, workers, aggregation_method, cutoff, method, features, outcome, k)