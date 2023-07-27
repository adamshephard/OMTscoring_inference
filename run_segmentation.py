"""
Use TIAToolbox Multi-Task Segmentor to gt nuclear/layer segmentations with HoVer-Net+.
"""
import os
import glob
from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

input_wsi_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/wsis/"
output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output/"

wsi_file_list = glob.glob(input_wsi_dir + "*")

multi_segmentor = MultiTaskSegmentor(
    pretrained_model="hovernetplus-oed",
    num_loader_workers=10,
    num_postproc_workers=10,
    batch_size=8,
    auto_generate_mask=False,
)

# WSI prediction
wsi_output = multi_segmentor.predict(
    imgs=wsi_file_list,
    masks=None,
    save_dir=os.path.join(output_dir, "hovernetplus"),
    mode="wsi",
    on_gpu=True,
    crash_on_exception=True,
)

