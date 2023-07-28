import os
import numpy as np
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.utils.data as data
import torch


class featuresdataset_inference(data.Dataset):
    def __init__(self, data_path, data_frame, raw_images=False, transform=None):
        # opens data dictionary (lib)
        self.data_path = data_path
        self.data_frame = data_frame
        list_of_images = data_frame['wsi'].tolist()
        list_of_cohorts = data_frame['cohort'].tolist()
        tiles = []
        ntiles = []
        slideIDX = []
        targets = []
        slides = []
        cohorts = []
        j = 0
        for i, case in enumerate(list_of_images):
            cohort = list_of_cohorts[i]
            label = data_frame.loc[(data_frame['wsi']==case) & (data_frame['cohort']==cohort)]['label'].item()
            t = []
            path = os.path.join(self.data_path, case)

            # if os.path.exists(path):
            for f in os.listdir(path):
                t.append(os.path.join(path, f)) #paths to all tiles of the slide
            slides.append(path)          # slide path
            tiles.extend(t)              # all tiles of the data set
            ntiles.append(len(t))        # number of tiles in this slide
            slideIDX.extend([j]*len(t))  # array of index --- all tiles of same slide gets unique index
            targets.append(label)        # slide groudn truth label as the tile ground truth label
            cohorts.append(cohort)       # some slides are repated in cohorts, this locates the correct slide
            j+=1
        print('Number of Slides: {}'.format(len(targets)))
        print('Number of tiles: {}'.format(len(tiles)))
        print('Max tiles ', max(ntiles))
        print('Min tiles', min(ntiles))
        print('Average tiles: ', np.mean(ntiles))
        self.slideIDX = slideIDX
        self.ntiles = ntiles
        self.tiles = tiles
        self.targets = targets
        self.slides = slides
        self.mode = None
        self.transform = transform
        self.raw_images = raw_images
        self.cohorts = cohorts

    def __getitem__(self,index):
        tile = self.tiles[index]
        case = os.path.basename(tile).split('_')[0]
        slideIDX = self.slideIDX[index]
        target = self.targets[slideIDX]
        if self.raw_images is True:
            img = Image.open(str(tile)).convert('RGB')
            img = img.resize((256,256),Image.BILINEAR)  
            if self.transform is not None:
                img = self.transform(img)
        else:                  
            img = torch.load(str(tile))
        return [img, (target, case)]

    def __len__(self):
        return len(self.tiles)

class featuresdataset_wsi_inference(data.Dataset):
    def __init__(self, patches, rois, layer_col_dict, raw_images=False, transform=None):
        # opens data dictionary (lib)
        self.color_dict = layer_col_dict
        self.patches = patches
        self.rois = rois
        self.transform = transform
        self.raw_images = raw_images

    def __getitem__(self,index):
        tile = self.patches[index]
        coords = torch.from_numpy(self.rois[index])
        if self.raw_images is True:
            img = Image.fromarray(tile)
            img = img.resize((256,256),Image.BILINEAR)  
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = torch.from_numpy(tile)
        return [img, coords]

    def __len__(self):
        return len(self.patches)