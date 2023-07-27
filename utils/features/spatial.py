import itertools
from collections import OrderedDict
from typing import Callable, Mapping, Union

import numpy as np
from sklearn.neighbors import KDTree

from utils.features.abc import ABCFeatures
from sklearn import neighbors

class KNNFeatures(ABCFeatures):
    def __init__(
            self,
            unique_type_list: Union[list, np.ndarray],
            pair_list: Union[list, np.ndarray] = None,
            kdtree: neighbors.KDTree = None,
            compute_basic_stats: Callable[[Union[list, np.ndarray]],
                                          Mapping[str, str]] = None,
            ):
        """
        `unique_type_list`: will be used to generate pair of typing for
            computing colocalization. This will explode if there are too
            many !
        """
        super().__init__()
        self.kdtree = kdtree
        self.pair_list = pair_list
        self.type_list = None  # holder for runtime
        self.unique_type_list = unique_type_list
        if compute_basic_stats is not None:
            self._compute_basic_stats = compute_basic_stats
        self.state = {}
        return

    def _get_neighborhood_stat(self, neighbor_idx_list):
        nn_type_list = self.type_list[neighbor_idx_list]
        (
            unique_type_list,
            nn_type_frequency,
        ) = np.unique(nn_type_list, return_counts=True)
        # repopulate for information wrt provided type because the
        # neighborhood may not contain all possible types within the
        # image/subject/dataset
        unique_type_freqency = np.zeros(len(self.unique_type_list))
        unique_type_freqency[unique_type_list] = nn_type_frequency
        return unique_type_freqency

    def transform(
            self,
            pts_list: np.ndarray,
            type_list: np.ndarray,
            radius: float):
        """Extract feature from given nuceli list.

        During calculation, any node that has no neighbor will be
        excluded from calculating summarized statistics.

        `pts_list`: Nx2 array of coordinates. In case `kdtree` is provided
            when creating the object, `pts_list` input here must be the same
            one used to construct `kdtree` else the results will be bogus.
        `type_list`: N array indicating the type at each coordinate.

        """
        self.type_list = type_list
        kdtree = self.kdtree
        if kdtree is None:
            kdtree = KDTree(pts_list, metric='euclidean')
        knn_list = kdtree.query_radius(
                        pts_list, radius,
                        return_distance=True,
                        sort_results=True)
        # each neighbor will contain self, so must remove self idx later
        knn_list = list(zip(*knn_list))

        self.state = {}
        # each clique is the neigborhood within the `radius` of point at
        # idx within `pts_list`
        keep_idx_list = []
        nn_raw_freq_list = []  # counnting occurence
        for idx, clique in enumerate(knn_list):
            nn_idx, nn_dst = clique  # index, distance
            if len(nn_idx) <= 1:
                continue
            # remove self to prevent bogus
            sel = nn_idx != idx
            nn_idx = nn_idx[sel]
            nn_dst = nn_dst[sel]

            nn_type_freq = self._get_neighborhood_stat(nn_idx)
            nn_raw_freq_list.append(nn_type_freq)
            keep_idx_list.append(idx)
            self.state[idx] = {
                'idx': idx,
                'nn_idx': nn_idx,
                'nn_dst': nn_dst,
                'nn_type_freq': nn_type_freq
            }
        keep_idx_list = np.array(keep_idx_list)

        # from counting to ratio, may not entirely make sense
        # on the entire dataset
        def count_to_frequency(a):
            """Helper to make func less long."""
            return a / (np.sum(a, keepdims=True, axis=-1) + 1.0e-8)

        nn_raw_freq_list = np.array(nn_raw_freq_list)
        # nn_rat_freq_list = count_to_frequency(nn_raw_freq_list)

        # get summarized occurence of pair of types, given the source
        # and its neighborhood
        pair_list = self.pair_list
        if pair_list is None:
            pair_list = itertools.product(self.unique_type_list, repeat=2)
            pair_list = list(pair_list)

        stat_dict = OrderedDict()
        type_list = self.type_list[keep_idx_list]
        for src_type, nn_type in pair_list:
            sel = type_list == src_type
            if np.any(sel):
                src_nn_raw_stat_list = nn_raw_freq_list[sel][:, nn_type]
                # src_nn_frq_stat_list = nn_rat_freq_list[sel][:, nn_type]
                # now get basic statistics
                raw_stat = self._compute_basic_stats(src_nn_raw_stat_list)
                # rat_stat = self._compute_basic_stats(src_nn_frq_stat_list)
            else:
                # incase src_type doesnt exist in this subject, return dummy
                # np.nan for fill in/placeholder
                raw_stat = self._compute_basic_stats(np.full(10, np.nan))
                # rat_stat = self._compute_basic_stats(np.full(10, np.nan))

            # ! may not be sustainable, may need to split into
            # ! [action description] [group description] [origin description]
            desc = f'Number of [{nn_type}] surrounding [{src_type}] within {radius}'
            raw_stat = {f'[{k}] {desc}': v for k, v in raw_stat.items()}
            stat_dict.update(raw_stat)

            # on the hindsight, rat_stat at this stage doesnt make sense
            # for example 1/1 inflammatory is no different from 8/8
            # desc = f'Percentage of [{nn_type}] surrounding [{src_type}] within {radius}'
            # rat_stat = {f'[{k}] {desc}': v for k, v in rat_stat.items()}
            # stat_dict.update(rat_stat)
        return stat_dict

def get_spatial_features(dict, nr_types):
    nuc_com = []
    nuc_type = []
    for nuc_id in dict:
        nuc_t = dict[nuc_id]['type']
        if nuc_t == 5:
            nuc_t = 1
        if nuc_t in [2,3,4]:
            nuc_t = 2
        nuc_type.append(nuc_t-1)
        nuc_com.append(dict[nuc_id]['centroid'])
    nuc_com = np.asarray(nuc_com)
    nuc_type = np.asarray(nuc_type)
    sel = nuc_type != -1  # this is other expressions
    nuc_com = nuc_com[sel]
    nuc_type = nuc_type[sel]
    nuc_com = nuc_com * 1.0
    uid_comb = np.arange(0, nr_types)
    mpp = 0.5
    num_step = 4#8
    max_radius_mpp = 200  # in mpp
    unit_radius_mpp = max_radius_mpp / num_step
    unit_radius_pix = np.ceil(unit_radius_mpp / mpp)
    fdict = OrderedDict()
    radius_list = unit_radius_pix * np.arange(1, num_step+1)
    kdtree = KDTree(nuc_com)
    for radius in radius_list:
        fxtor = KNNFeatures(
                    kdtree=kdtree,
                    pair_list=None,
                    unique_type_list=uid_comb)
        stats = fxtor.transform(nuc_com, nuc_type, radius=radius)
        fdict.update(stats)
    return fdict
# %%
