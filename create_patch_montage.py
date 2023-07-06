"""
This script is to create a montage of top n predicted patches from IDARS. Also performs nuclear analyses.
"""
import os
import glob

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 24#12
plt.rcParams["figure.figsize"] = (20,20)
import skimage.util
import random

from tiatoolbox.utils.misc import imread, imwrite

ARG_DICT = {
    'tp': 'True Positives',
    'fp': 'False Positives',
    'tn': 'True Negatives',
    'fn': 'False Negatives',
}

def create_montage(cases, arg, patch_list, output_folder, outcome, repeat, nr_top_patches, nr_patches):
    patches_montage = []
    for case in cases:
        patches = []
        for patch in patch_list:
            if case in patch:
                patches.append(patch)
        for patch_file in sorted(patches)[:nr_top_patches]:
            patch = imread(patch_file)
            patches_montage.append(patch)
    patches_montage_shuffled = patches_montage.copy()
    random.shuffle(patches_montage_shuffled)
    montage = skimage.util.montage(patches_montage_shuffled[:nr_patches], multichannel=True, grid_shape=(7,7))
    plt.imshow(montage)
    plt.axis('off')
    plt.title(f'{outcome.capitalize()} Repeat {repeat}: {ARG_DICT[arg]} - Top {nr_top_patches} Patches')
    # plt.show(block=False)
    plt.savefig(os.path.join(output_folder, f'{outcome}_repeat-{repeat}_{arg}.png'))
    return


input_root = '/data/ANTICIPATE/outcome_prediction/MIL/idars/output/cohort_1,3,4/transformation/mlp/morph_features_104_64ftrs_100eps/oed/' #resnet34/raw_images/oed/'
outcome = 'transformation'
nr_repeats = 3
nr_folds = 5

nr_top_patches = 5 # how mnay top patches to use
nr_patches = 49

output_folder = os.path.join(input_root, 'patch_montages')
os.makedirs(output_folder, exist_ok=True)

for repeat in range(1, nr_repeats+1):
    # input_csv = os.path.join(input_root, 'summary_predictions.csv')
    top_patch_list = []
    bottom_patch_list = []
    for fold in range(nr_folds):
        # input_patch_dir = os.path.join(input_root, f'repeat_{repeat}/best{fold}', 'top_5_patches')
        # for patch in glob.glob(input_patch_dir+'/*/*.png'):
        #     patch_list.append(patch)
    # results_df = pd.read_csv(input_csv)
        top_input_patch_dir = os.path.join(input_root, f'repeat_{repeat}/best{fold}', 'top_5_patches')
        m_data = pd.read_csv(os.path.join(input_root, f'repeat_{repeat}/best{fold}/predictions.csv'))
        for patch in glob.glob(top_input_patch_dir+'/*/*/*.png'):
            top_patch_list.append(patch)
            coh = os.path.basename(os.path.dirname(os.path.dirname(patch))).split('_')[1]
        bottom_input_patch_dir = os.path.join(input_root, f'repeat_{repeat}/best{fold}', 'bottom_5_patches')
        for patch in glob.glob(bottom_input_patch_dir+'/*/*/*.png'):
            bottom_patch_list.append(patch)
            coh = os.path.basename(os.path.dirname(os.path.dirname(patch))).split('_')[1]
        if fold == 0:
            results_df = m_data
        else:
            results_df = pd.concat([results_df, m_data])
    results_df = results_df.reset_index(drop=True)
    results_df = results_df.rename(columns={'y_pred':f'y_pred_repeat{repeat}'})
    results_df = results_df[['case', 'y_true', f'y_pred_repeat{repeat}']]
    cases_tps = results_df.loc[(results_df['y_true'] == 1) & (results_df[f'y_pred_repeat{repeat}'] == 1)]['case'].to_list() # True Positives
    cases_fps = results_df.loc[(results_df['y_true'] == 0) & (results_df[f'y_pred_repeat{repeat}'] == 1)]['case'].to_list() # False Positives
    cases_fns = results_df.loc[(results_df['y_true'] == 1) & (results_df[f'y_pred_repeat{repeat}'] == 0)]['case'].to_list() # False Negatives
    cases_tns = results_df.loc[(results_df['y_true'] == 0) & (results_df[f'y_pred_repeat{repeat}'] == 0)]['case'].to_list() # True Negatives

    if len(cases_tps) == 0:
        print(f'No TP cases for repeat {repeat}')
    else:
        create_montage(cases_tps, 'tp', top_patch_list, output_folder, outcome, repeat, nr_top_patches, nr_patches)
    if len(cases_fps) == 0:
        print(f'No FP cases for repeat {repeat}')
    else:
        create_montage(cases_fps, 'fp', bottom_patch_list, output_folder, outcome, repeat, nr_top_patches, nr_patches)
    if len(cases_fns) == 0:
        print(f'No FN cases for repeat {repeat}')
    else:
        create_montage(cases_fns, 'fn', bottom_patch_list, output_folder, outcome, repeat, nr_top_patches, nr_patches)
    if len(cases_tns) == 0:
        print(f'No TN cases for repeat {repeat}')
    else:
        create_montage(cases_tns, 'tn', top_patch_list, output_folder, outcome, repeat, nr_top_patches, nr_patches)