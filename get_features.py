"""
Generate morphological/spatial features for MLP (based on HoVer-Net+ output) for OMTscoring.
"""

import os
import glob
from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

input_wsi_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/wsis/"
hovernetplus_output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/hovernetplus/"
feature_output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/features/"
