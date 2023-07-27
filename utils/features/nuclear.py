import numpy as np
import pandas as pd

from utils.features.spatial import get_spatial_features
from utils.features.morph import get_inst_stat


def update_with_affix(dict, name):
    new_dict = {}
    for k in dict.keys():
        new_dict[name + ": " + k] = dict[k]
    return new_dict

def get_morph_features(dict, patch):
    # e.g. to consolidate pannuke types into 2 classes (other and epithelial)
    nuc_types = []
    stat_dict = {}
    for nuc_id in dict:
        nuc_type = dict[nuc_id]['type']
        if nuc_type == 5:
            nuc_type = 1
        if nuc_type in [2,3,4]:
            nuc_type = 2
        nuc_types.append(nuc_type)
        _, nuc_ftr_dict = get_inst_stat(nuc_id, dict[nuc_id], patch, nuc_type)
        stat_dict[nuc_id] = nuc_ftr_dict
    return stat_dict, nuc_types

def get_nuc_features(nuc_dict, patch_name, nr_types):
    blank_spatial_df = pd.read_csv("./utils/features/spatial_2class.csv", index_col=0, names=['values'])
    blank_morph_df = pd.read_csv("./utils/features/morph_2class.csv", index_col=0, names=['values'])
    blank_spatial_df = blank_spatial_df.to_dict()['values']
    blank_morph_dict = blank_morph_df.to_dict()['values']
    # first check for nuc data
    if len(nuc_dict) == 0:
        # Then no nuclei present - return empty dict of features
        # TODO!!!s
        morph_df = blank_morph_df
        spatial_df = blank_spatial_df
    else:
        try:
            spatial_dict_pre = get_spatial_features(nuc_dict, nr_types=nr_types)
        except:
            print(f"graph exception in patch: {patch_name}")
            spatial_dict_pre = blank_spatial_df.copy()
        # sanity check to make sure all fields are populated else make 0
        # also make any nans 0
        spatial_dict = blank_spatial_df.copy()
        for k, v in spatial_dict.items():
            if k in spatial_dict_pre:
                if np.isnan(spatial_dict_pre[k]):
                    spatial_dict[k] = int(0)
                else:
                    spatial_dict[k] = spatial_dict_pre[k]    
        morph_dict, _ = get_morph_features(nuc_dict, patch_name)
        morph_dframe = pd.DataFrame(morph_dict).transpose()
        feature_list = morph_dframe.columns[3:]
        type_list = morph_dframe.loc[:,'type'].to_numpy()
        sub_dframe = morph_dframe[feature_list]
        adict = {}
        def get_summary(df, nuc_type):
            sub_df = df.iloc[type_list == nuc_type]
            adict.update(update_with_affix(sub_df.mean().to_dict()  , 'type=%d-mu'  % nuc_type))
            adict.update(update_with_affix(sub_df.std().to_dict()   , 'type=%d-va'  % nuc_type))
            adict.update(update_with_affix(sub_df.min().to_dict()   , 'type=%d-min' % nuc_type))
            adict.update(update_with_affix(sub_df.max().to_dict()   , 'type=%d-max' % nuc_type))
        # need type conversion, else wont work with pandas quantile
        sub_dframe = sub_dframe.astype(np.float64)            
        for type_id in np.unique(type_list):
            get_summary(sub_dframe, type_id)
        # sanity check to make sure all fields are populated else make 0
        # also make any nans 0
        morph_dict = blank_morph_dict.copy()
        for k, v in morph_dict.items():
            if k in adict:
                if np.isnan(adict[k]):
                    morph_dict[k] = int(0)
                else:
                    morph_dict[k] = adict[k]  
        
        spatial_df = pd.DataFrame.from_dict(spatial_dict, orient='index')
        morph_df = pd.DataFrame.from_dict(morph_dict, orient='index')
        return morph_df, spatial_df

