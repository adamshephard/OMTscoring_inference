# IDaRS (LDH)
import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, f1_score, precision_recall_curve, average_precision_score
from scipy.stats.mstats import gmean


def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

def cal_f1_score(targets, prediction, cutoff):
    prediction = np.array(prediction)
    targets = np.array(targets)
    m1 = np.mean(prediction * targets)
    m2 = np.mean(prediction * (1-targets))
    cutoff1 = (m1 + m2) / 2.
    f1score1 = f1_score(targets, [1 if v >= cutoff1 else 0 for v in prediction])
    f1score2 = f1_score(targets, [1 if v >= cutoff else 0 for v in prediction])
    return f1score1, f1score2

def calc_metrics(target, prediction):
    fpr, tpr, thresholds = roc_curve(target, prediction)
    cutoff = cutoff_youdens_j(fpr, tpr, thresholds)
    roc_auc = auc(fpr, tpr)
    f1score, f1score2 = cal_f1_score(target, prediction, cutoff)
    precision, recall, _ = precision_recall_curve(target, prediction)
    average_precision = average_precision_score(target, prediction)
    return average_precision, roc_auc

def calculate_accuracy(output, target):
    preds = output.max(1, keepdim=True)[1]
    correct = preds.eq(target.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

# getting top k indices for each slide in the data set
def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

#different aggregation methods using the probabilities of each tile
def compute_aggregated_probabilities(group, data, k=10):
    wsi_dict = {}
    for idx, g in enumerate(group):
        g_id = wsi_dict.get(g, -1)
        if g_id == -1:
            wsi_dict[g] = [data[idx]]
        else:
            temp_data = wsi_dict[g]
            temp_data.append(data[idx])
            wsi_dict[g] = temp_data
    avg_p = []
    max_p = []
    sum_p = []
    md_p = []
    gm_p = []
    top_p = []
    md_vt = []
#     topk_p = []
    for each_wsi in wsi_dict.keys():
        wsi_predictions = wsi_dict[each_wsi]
        wsi_predictions = np.array(wsi_predictions, dtype='float64')
        avg_p.append(np.mean(wsi_predictions))
        max_p.append(np.max(wsi_predictions))
        sum_p.append(np.sum(wsi_predictions)) 
        md = np.median(wsi_predictions)
        md_p.append(md)
        gm_p.append(gmean(wsi_predictions))
        top_p.append(np.mean(wsi_predictions[wsi_predictions>=md]))
    sum_p = np.array(sum_p, dtype='float64')
    avg_p = np.array(avg_p, dtype='float64')
    max_p = np.array(max_p, dtype='float64')
    md_p = np.array(md_p, dtype='float64')
    gm_p = np.array(gm_p, dtype='float64')
    top_p = np.array(top_p, dtype='float64')
    return avg_p, max_p, sum_p, md_p, gm_p, top_p

# Get top k predicted patches from each slide
def get_topk_patches(group, tiles, data, k=10):
    topk_p = []
    wsi_patch_dict = {}
    wsi_prob_dict = {}
    for idx, g in enumerate(group):
        g_id = wsi_prob_dict.get(g, -1)
        if g_id == -1:
            wsi_prob_dict[g] = [data[idx]]
            wsi_patch_dict[g] = [tiles[idx]]
        else:
            temp_pred_data = wsi_prob_dict[g]
            temp_tiles_data = wsi_patch_dict[g]
            temp_pred_data.append(data[idx])
            temp_tiles_data.append(tiles[idx])
            wsi_prob_dict[g] = temp_pred_data
            wsi_patch_dict[g] = temp_tiles_data
    for each_wsi in wsi_prob_dict.keys():
        wsi_predictions = wsi_prob_dict[each_wsi]
        wsi_tiles = np.array(wsi_patch_dict[each_wsi])
        wsi_predictions = np.array(wsi_predictions, dtype='float64')
        topk_ind = sorted(range(len(wsi_predictions)), key=lambda i: wsi_predictions[i])[-k:]
        topk_p.append(list(wsi_tiles[topk_ind]))
    # topk_p = np.array(topk_p)
    return topk_p

# Get bottom k predicted patches from each slide
def get_bottomk_patches(group, tiles, data, k=10):
    bottomk_p = []
    wsi_patch_dict = {}
    wsi_prob_dict = {}
    for idx, g in enumerate(group):
        g_id = wsi_prob_dict.get(g, -1)
        if g_id == -1:
            wsi_prob_dict[g] = [data[idx]]
            wsi_patch_dict[g] = [tiles[idx]]
        else:
            temp_pred_data = wsi_prob_dict[g]
            temp_tiles_data = wsi_patch_dict[g]
            temp_pred_data.append(data[idx])
            temp_tiles_data.append(tiles[idx])
            wsi_prob_dict[g] = temp_pred_data
            wsi_patch_dict[g] = temp_tiles_data
    for each_wsi in wsi_prob_dict.keys():
        wsi_predictions = wsi_prob_dict[each_wsi]
        wsi_tiles = np.array(wsi_patch_dict[each_wsi])
        wsi_predictions = np.array(wsi_predictions, dtype='float64')
        bottomk_ind = sorted(range(len(wsi_predictions)), key=lambda i: wsi_predictions[i])[:k]
        bottomk_p.append(list(wsi_tiles[bottomk_ind]))
    # topk_p = np.array(topk_p)
    return bottomk_p

# majority vote aggregations using the predicted class of tiles
def compute_aggregated_predictions(group, data):
    wsi_dict = {}
    for idx, g in enumerate(group):
        g_id = wsi_dict.get(g, -1)
        if g_id == -1:
            wsi_dict[g] = [data[idx]]
        else:
            temp_data = wsi_dict[g]
            temp_data.append(data[idx])
            wsi_dict[g] = temp_data
    pos_pred = []
    sum_p = []
    for each_wsi in wsi_dict.keys():
        wsi_predictions = wsi_dict[each_wsi]
        # total number of positive predition
        sum_of_wsi_pos_pred = np.sum(wsi_predictions)
        # majority vote
        mj_vt = sum_of_wsi_pos_pred/len(wsi_predictions)
        sum_p.append(sum_of_wsi_pos_pred) 
        pos_pred.append(mj_vt) 
    sum_p = np.array(sum_p, dtype='float64')
    pos_pred = np.array(pos_pred, dtype='float64')
    return pos_pred, sum_p

#average of top 10 tiles aggregation
def group_avg_df(groups, data):
# intialise data of lists.
    dfra = {'Slide':groups,'value':data}
# Create DataFrame
    df = pd.DataFrame({'Slide':groups,'value':data})
    group_average_df = df.groupby('Slide')['value'].apply(lambda grp: grp.nlargest(10).mean())
    group_average = group_average_df.tolist()
    return group_average 

#average of top 50 tiles aggregation
def group_avg_df_50(groups, data):
# intialise data of lists.
    dfra = {'Slide':groups,'value':data}
# Create DataFrame
    df = pd.DataFrame({'Slide':groups,'value':data})
    group_average_df = df.groupby('Slide')['value'].apply(lambda grp: grp.nlargest(50).mean())
    group_average = group_average_df.tolist()
    return group_average