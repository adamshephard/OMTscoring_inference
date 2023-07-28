
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from typing import Union, Mapping, Callable

class ABCFeatures(ABC):
    """
    """
    def __init__(
            self,
            compute_basic_stats: Callable[[Union[list, np.ndarray]],
                                          Mapping[str, str]] = None
            ):

        if compute_basic_stats is not None:
            self._compute_basic_stats = compute_basic_stats
        return

    def _compute_basic_stats(
                self,
                val_list: Union[list, np.ndarray]
            ) -> Mapping[str, float]:

        val_list = np.array(val_list)
        stat_dict = OrderedDict()
        stat_dict['Max'] = np.max(val_list)
        stat_dict['Min'] = np.min(val_list)
        stat_dict['Mean'] = np.mean(val_list)
        stat_dict['Std'] = np.std(val_list)
        return stat_dict
