import os
import glob
from tiatoolbox.utils.misc import imread, imwrite
import numpy as np
from tqdm import tqdm
from utils.utils import get_heatmap

nr_repeats = 3
heatmap_dir = '/data/ANTICIPATE/outcome_prediction/MIL/idars/output/cohort_1,3,4/transformation/mlp/morph_features_104_64ftrs_100eps/oed/heatmaps/'#'/data/ANTICIPATE/outcome_prediction/MIL/idars/output/train-sheffield_test-belfast,birmingham,brazil/transformation/resnet34_SA2/raw_images_30eps/oed/heatmaps/'#'/data/ANTICIPATE/outcome_prediction/MIL/idars/output/cohort_1,3,4/transformation/resnet34/raw_images/heatmaps/'

out_dir = os.path.join(heatmap_dir, 'merged')
os.makedirs(out_dir, exist_ok=True)

for case_path in tqdm(sorted(glob.glob(heatmap_dir+'/repeat_1/*npy'))):
    case = os.path.basename(case_path).split('.')[0]
    if os.path.exists(os.path.join(out_dir, case+".png")):
        print(f'Skipping case {case} as merged heatmap laready exists.')
        continue
    heatmap = np.load(case_path)
    count = 1
    try:
        for repeat in range(2, nr_repeats+1):
            if os.path.exists(os.path.join(heatmap_dir, f'repeat_{repeat}/', case+'.npy')):
                heatmap_2 = np.load(os.path.join(heatmap_dir, f'repeat_{repeat}/', case+'.npy'))
                heatmap += heatmap_2
                count += 1
        if count == 3:
            heatmap = heatmap / 3
            heatmap_col = get_heatmap(heatmap)
            imwrite(os.path.join(out_dir, case+".png"), heatmap_col)
            np.save(os.path.join(out_dir, case+".npy"), heatmap)
    except:
        print(f'Failed for case: {case}')