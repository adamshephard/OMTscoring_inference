"""
Use TIAToolbox Multi-Task Segmentor to gt nuclear/layer segmentations with HoVer-Net+.
"""
import os
import glob
import shutil
from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

input_wsi_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/wsis/"
output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output2/"

wsi_file_list = glob.glob(input_wsi_dir + "*")[1:]

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
    save_dir=os.path.join(output_dir, "hovernetplus/tmp"),
    mode="wsi",
    on_gpu=True,
    crash_on_exception=True,
)


# Rename TIAToolbox output files to readability
layer_dir = os.path.join(output_dir, "hovernetplus", "layers")
nuclei_dir = os.path.join(output_dir, "hovernetplus", "nuclei")
os.makedirs(layer_dir, exist_ok=True)
os.makedirs(nuclei_dir, exist_ok=True)

for out in enumerate(wsi_output):
    basename = os.path.basename(out[0]).split(".")[0]
    outname = os.path.basename(out[1]).split(".")[0]
    shutil.move(
        os.path.join(output_dir, "hovernetplus/tmp", f"{outname}.1.npy"),
        os.path.join(layer_dir, basename + ".npy"),
        )   
    shutil.move(
        os.path.join(output_dir, "hovernetplus/tmp", f"{outname}.0.dat"),
        os.path.join(nuclei_dir, basename + ".dat"),
        )



