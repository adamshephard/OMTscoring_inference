import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, f1_score, precision_recall_curve, average_precision_score, RocCurveDisplay, plot_roc_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from viz_utils.auroc_curve import auroc_curve

from utils.metrics import cutoff_youdens_j, cal_f1_score


def calc_metrics(target, prediction):
    fpr, tpr, thresholds = roc_curve(target, prediction)
    cutoff = cutoff_youdens_j(fpr, tpr, thresholds)
    roc_auc = auc(fpr, tpr)
    f1score, f1score2 = cal_f1_score(target, prediction, cutoff)
    precision, recall, _ = precision_recall_curve(target, prediction)
    average_precision = average_precision_score(target, prediction)
    # return average_precision, roc_auc
    return f1score2, roc_auc, cutoff

def calculate_accuracy(output, target):
    preds = output.max(1, keepdim=True)[1]
    correct = preds.eq(target.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

#function to plot metrics
def plot_metrics(target, prediction, set, fold_name):
    fpr, tpr, thresholds = roc_curve(target, prediction)
    roc_auc = auc(fpr, tpr)
    print('roc_auc is:', roc_auc)
    precision, recall, _ = precision_recall_curve(target, prediction)
    average_precision = average_precision_score(target, prediction)
    pr_auc = auc(recall, precision)
    print('Average precision-recall score: {0:0.2f}'.format(
         average_precision))
    plt.figure(figsize=(12, 4))
    lw = 2
    plt.subplot(121)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label=f'{fold_name} ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve, AUC={0:0.2f}'.format(roc_auc))
    plt.legend(loc="lower right")
    plt.subplot(122)
    plt.step(recall, precision, alpha=0.4, color='darkorange', where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='navy', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1.05])
    plt.xlim([0, 1])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    # plt.savefig(os.path.join(output, 'roc_pr' + set + '.png'))
    # plt.close(plt.gcf())
    return


label_dict ={
    'avg prob': 'Avg Probability',
    'max prob': 'Max Probability',
    'sump': 'Sum Probability',
    'med': 'Med Probability',
    'gmp': 'GM Probability',
    'avgtop': 'AvgTop Probability',
    'top5': 'A10 Probability',
    'mj vote raw': 'Raw Probability',
    'mj vote': 'MjVt Probability',
}

outcome = 'transformation' #'recurrence' #'transformation'
features = 'raw_images_30eps' #'morph_features_104_64ftrs_100eps' #'raw_images' #'deep_features' #'morph_features_156ftrs'
method = 'resnet34_SA2' #'mlp' #'resnet34_SA' #'mlp' #'resnet34' #'fc'
input_dir = f'/data/ANTICIPATE/outcome_prediction/MIL/idars/output/cohort_1,3,4/{outcome}/{method}/{features}/oed/'

nr_repeats = 3
nr_folds = 5
count = 0

f1_data = pd.DataFrame(columns=[
    'repeat', 'fold', 'mj vote', 'mj vote raw', 'avg prob',
    'max prob', 'sump', 'top5', 'med', 'gmp', 'avgtop'
    ])
auc_data = pd.DataFrame(columns=[
    'repeat', 'fold', 'mj vote', 'mj vote raw', 'avg prob',
    'max prob', 'sump', 'top5', 'med', 'gmp', 'avgtop'
    ])  
cutoff_data = pd.DataFrame(columns=[
    'repeat', 'fold', 'mj vote', 'mj vote raw', 'avg prob',
    'max prob', 'sump', 'top5', 'med', 'gmp', 'avgtop'
    ])     
for repeat in range(1, nr_repeats+1):
    for fold in range(nr_folds):
        # try:
        data_file = os.path.join(input_dir, f'repeat_{repeat}/best{fold}/prob_test_slides.csv')
        # fold_data = pd.read_csv(data_file)
        # if count == 0:
        #     data = fold_data
        # else:
        #     data = data.append(fold_data)
        #     data = data.reset_index(drop=True)
        # count += 1
        fold_data = pd.read_csv(data_file)
        data = fold_data
        test_slide_avg = data['Avg Probability'].tolist()
        test_slide_max = data['Max Probability'].tolist()
        test_slide_sum = data['Sum Probability'].tolist()
        test_slide_md = data['Med Probability'].tolist()
        test_slide_gm = data['GM Probability'].tolist()
        test_slide_avgtop = data['AvgTop Probability'].tolist()
        test_slide_avgt5 = data['A10 Probability'].tolist()
        test_slide_mjvt_raw = data['Raw Probability'].tolist()
        test_slide_mjvt = data['MjVt Probability'].tolist()
        targets = data['Targets'].tolist()
        f1_mv, roc_auc_mjvt, cutoff_mjvt = calc_metrics(targets, test_slide_mjvt)
        f1_mv_r, roc_auc_mjv_raw, cutoff_mjv_raw = calc_metrics(targets, test_slide_mjvt_raw)
        f1_md, roc_auc_mdp, cutoff_mdp = calc_metrics(targets, test_slide_md)
        f1_gp, roc_auc_gmp, cutoff_gmp = calc_metrics(targets, test_slide_gm)
        f1_ap, roc_auc_avgp, cutoff_avgp = calc_metrics(targets, test_slide_avg)
        f1_mp, roc_auc_maxp, cutoff_maxp = calc_metrics(targets, test_slide_max)
        f1_sp, roc_auc_sump, cutoff_sump = calc_metrics(targets, test_slide_sum)
        f1_5, roc_auc_top5, cutoff_top5 = calc_metrics(targets, test_slide_avgt5)
        f1_at, roc_auc_avgtop, cutoff_avgtop = calc_metrics(targets, test_slide_avgtop)
        f1_data = f1_data.append({
            'repeat': int(repeat),
            'fold': int(fold),
            'mj vote': f1_mv,
            'mj vote raw': f1_mv_r,
            'avg prob': f1_ap,
            'max prob': f1_mp,
            'sump': f1_sp,
            'top5': f1_5,
            'med': f1_md,
            'gmp': f1_gp,
            'avgtop': f1_at
            }, ignore_index=True)
        auc_data = auc_data.append({
            'repeat': int(repeat),
            'fold': int(fold),
            'mj vote': roc_auc_mjvt,
            'mj vote raw': roc_auc_mjv_raw,
            'avg prob': roc_auc_avgp,
            'max prob': roc_auc_maxp,
            'sump': roc_auc_sump,
            'top5': roc_auc_top5,
            'med': roc_auc_mdp,
            'gmp': roc_auc_gmp,
            'avgtop': roc_auc_avgtop
            }, ignore_index=True)
        cutoff_data = cutoff_data.append({
            'repeat': int(repeat),
            'fold': int(fold),
            'mj vote': cutoff_mjvt,
            'mj vote raw': cutoff_mjv_raw,
            'avg prob': cutoff_avgp,
            'max prob': cutoff_maxp,
            'sump': cutoff_sump,
            'top5': cutoff_top5,
            'med': cutoff_mdp,
            'gmp': cutoff_gmp,
            'avgtop': cutoff_avgtop
            }, ignore_index=True)
        # except:
        #     continue

f1_data['repeat'] = f1_data['repeat'].astype('int')
f1_data['fold'] = f1_data['fold'].astype('int')
df_mean = f1_data.mean()
df_std = f1_data.std()
f1_data.loc['mean'] = df_mean
f1_data.loc['std'] = df_std
f1_data['repeat']['mean'] = "-"
f1_data['repeat']['std'] = "-"
f1_data['fold']['mean'] = "-"
f1_data['fold']['std'] = "-"   
auc_data['repeat'] = auc_data['repeat'].astype('int')
auc_data['fold'] = auc_data['fold'].astype('int')
df_mean = auc_data.mean()
df_std = auc_data.std()
auc_data.loc['mean'] = df_mean
auc_data.loc['std'] = df_std
auc_data['repeat']['mean'] = "-"
auc_data['repeat']['std'] = "-"
auc_data['fold']['mean'] = "-"
auc_data['fold']['std'] = "-"   

# print('AUC')
# print(auc_data)
# print('F1')
# print(f1_data)

auc_data.to_csv(os.path.join(input_dir, 'auroc_summary.csv'))
f1_data.to_csv(os.path.join(input_dir, 'f1_summary.csv'))
cutoff_data.to_csv(os.path.join(input_dir, 'cutoff_summary.csv'))


vals = auc_data.drop(columns=['repeat', 'fold']).loc['mean'].values.tolist()
labels = auc_data.drop(columns=['repeat', 'fold']).loc['mean'].keys().tolist()
top_label = labels[vals.index(max(vals))]


# AUROC curve
fprs = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
f1s = []
summary_data = pd.DataFrame(columns=[
    'repeat', 'fold', 'f1', 'auroc',
    ])

preds = []
targets = []
for repeat in range(1, nr_repeats+1):
    for fold in range(nr_folds):
        try:
            data_file = os.path.join(input_dir, f'repeat_{repeat}/best{fold}/prob_test_slides.csv')
            fold_data = pd.read_csv(data_file)
            data = fold_data
            pred = data[label_dict[top_label]].tolist()
            target = data['Targets'].tolist()
            # preds.append(pred)
            # targets.append(target)
            fpr, tpr, thresholds = roc_curve(target, pred)
            cutoff = cutoff_youdens_j(fpr, tpr, thresholds)
            f1score, f1score2 = cal_f1_score(target, pred, cutoff)
            roc_auc = auc(fpr, tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
            f1s.append(f1score2)
            summary_data = pd.concat([summary_data, pd.DataFrame.from_records([{
                'repeat': int(repeat),
                'fold': int(fold),
                'f1': f1score2,
                'auroc': roc_auc,
                'cutoff': cutoff,
                }])])
        except:
            continue

fig, ax = auroc_curve(tprs, mean_fpr, aucs)
# plt.show(block=False)
output_path = os.path.join(input_dir, 'auroc_curves.png')
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
summary_data.to_csv(os.path.join(input_dir, f'metrics_summary_{top_label}.csv'))
