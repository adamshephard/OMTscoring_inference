import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, f1_score, precision_recall_curve, average_precision_score, RocCurveDisplay, plot_roc_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt

from viz_utils.auroc_curve import auroc_curve


### Inputs Files and Paramaters ###
method = "pathologist"
features = "binary" # Choose from 'who', 'who_mildmod', 'who_mild', 'binary'. who_mildmod is mild/moderate classes combined
outcome = 'transformation' #'recurrence'
data_file = f'/data/data/ANTICIPATE/oed_data_train_test_new_corrected_260623.csv' #f'/data/data/ANTICIPATE/oed_data_train_test_new.csv' #f'/data/data/ANTICIPATE/sheffield/new_clinical_data_221222/sheffield_data_c1,3,4_{cohort}_5-folds.csv'
output = f'/data/ANTICIPATE/outcome_prediction/MIL/idars/output/train-sheffield_test-belfast,birmingham,brazil/{outcome}/{method}/{features}/' #f'/data/ANTICIPATE/outcome_prediction/MIL/idars/output/train-sheffield_test-birmingham,belfast_noqc/{outcome}/{method}/{features}/' #f'/data/ANTICIPATE/outcome_prediction/MIL/idars/output/cohort_1,3,4/{outcome}/{method}/{features}/'             # name of output file

if not os.path.exists(output):
    os.mkdir(output)

dataset = pd.read_csv(data_file)

# For who stratifications, we need to take the raw who grade and change the stratification
if features in ['who_mild', 'who_mildmod']:
    ftr = 'who'
else:
    ftr = features

test_data = dataset[dataset['test']==1][['slide_name', outcome, 'cohort', 'vendor', ftr]]
test_data = test_data.rename(columns={"slide_name": "wsi", outcome: "label"})

y_pred = test_data[ftr].astype('int').to_list()

if features == 'who_mildmod': 
    y_pred = [0 if y == 1 else y for y in y_pred]
    y_pred = [1 if y == 2 else y for y in y_pred]
if features == 'who_mild':
    y_pred = [1 if y == 1 else y for y in y_pred]
    y_pred = [1 if y == 2 else y for y in y_pred]

y_true = test_data['label'].to_list()
f1 = f1_score(y_true, y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
summary_data = pd.DataFrame({'f1': [f1], 'auroc': [roc_auc]})
summary_data.to_csv(os.path.join(output, f'metrics_summary.csv'))

# AUROC curve
fig, ax = auroc_curve(tpr, fpr, auc)
output_path = os.path.join(output, 'auroc_curves.png')
plt.savefig(output_path)

