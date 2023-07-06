import os
import numpy as np
import random
import pandas as pd
from sklearn.metrics import auc, roc_curve, f1_score, precision_recall_curve, average_precision_score, RocCurveDisplay, plot_roc_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean

from utils.metrics import group_avg_df, compute_aggregated_predictions, compute_aggregated_probabilities,group_argtopk,calculate_accuracy, calc_metrics
from viz_utils.auroc_curve import auroc_curve

# # some global variable
problem = 'oed'                                                              # prediction problem.
folds = 1#5                                                                    # number of fold to execute
repeats = 3                                                                     # number of times each fold runs
workers = 1                                                                 # help='number of data loading workers
test_every = 1                                                               # test on val every (default: 1
cohort = 'oed'
method = 'pathologist'
features = 'who_mild' #'binary'
outcome = 'transformation' #'recurrence'
data_file = f'/data/data/ANTICIPATE/oed_data_train_test_new_noqc.csv' #f'/data/data/ANTICIPATE/oed_data_train_test_new.csv' #f'/data/data/ANTICIPATE/sheffield/new_clinical_data_221222/sheffield_data_c1,3,4_{cohort}_5-folds.csv'
output = f'/data/ANTICIPATE/outcome_prediction/MIL/idars/output/train-sheffield_test-birmingham,belfast_noqc/{outcome}/{method}/{features}/' #f'/data/ANTICIPATE/outcome_prediction/MIL/idars/output/train-sheffield_test-belfast,birmingham,brazil/{outcome}/{method}/{features}/' #f'/data/ANTICIPATE/outcome_prediction/MIL/idars/output/cohort_1,3,4/{outcome}/{method}/{features}/'             # name of output file
external = True
subcohort = 5#'birmingham'

summary_data = pd.DataFrame(columns=[
    'repeat', 'fold', 'f1', 'auroc',
    ])

if not os.path.exists(output):
    # os.mkdir(output)     
    os.makedirs(output, exist_ok=True)   

output =  os.path.join(output, problem)
if not os.path.exists(output):
    os.mkdir(output)

temp_output = output

dataset = pd.read_csv(data_file)

fprs = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
f1s = []

if external == False:
    if features in ['who_mildmod', 'who_mild']:
        ftr = 'who'
    else:
        ftr = features
else:
    # ftr = 'who'
    if features == 'who_mildmod':
        ftr = 'who_compact'
    else:
        ftr = 'who'

for repeat in range(1, repeats+1):   # number of folds
    for fold in range(folds):      # number of runs for each fold
        if external == False:
            test_data = dataset[dataset[f'{outcome}_fold_{repeat}']==fold][['slide_name', outcome, 'cohort', 'vendor', ftr]]
        else:
            test_data = dataset[dataset['test']==1][['slide_name', outcome, 'cohort', 'vendor', ftr]]
        test_data = test_data.rename(columns={"slide_name": "wsi", outcome: "label"})
        # test_data = test_data.loc[test_data[ftr].notna()]
        # if subcohort is not None:
        #     test_data = test_data.loc[test_data['cohort']==5]

        y_pred = test_data[ftr].to_list()
        if external == False:
            if features == 'who_mildmod':
                y_pred = [0 if y == 1 else y for y in y_pred]
                y_pred = [1 if y == 2 else y for y in y_pred]
            elif features == 'who_mild':
                y_pred = [1 if y == 1 else y for y in y_pred]
                y_pred = [1 if y == 2 else y for y in y_pred]
        y_true = test_data['label'].to_list()
        f1 = f1_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        summary_data = summary_data.append({
            'repeat': int(repeat),
            'fold': int(fold),
            'f1': f1,
            'auroc': roc_auc,
            }, ignore_index=True)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        f1s.append(f1)

# AUROC curve
fig, ax = auroc_curve(tprs, mean_fpr, aucs)
# plt.show(block=False)
output_path = os.path.join(temp_output, 'auroc_curves.png')
plt.savefig(output_path)


summary_data = summary_data.reset_index(drop=True)
df_mean = summary_data.mean()
df_std = summary_data.std()
summary_data.loc['mean'] = df_mean
summary_data.loc['std'] = df_std
summary_data['repeat']['mean'] = "-"
summary_data['repeat']['std'] = "-"
summary_data['fold']['mean'] = "-"
summary_data['fold']['std'] = "-" 
summary_data.to_csv(os.path.join(temp_output, f'metrics_summary.csv'))
