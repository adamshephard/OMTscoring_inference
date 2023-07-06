import os
import numpy as np
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.utils.data as data
import torch
from tiatoolbox.utils.misc import imread, imwrite
from tiatoolbox.wsicore.wsireader import WSIReader, OpenSlideWSIReader, VirtualWSIReader, WSIMeta
from tiatoolbox.tools import patchextraction
from utils.utils import mask2epithmask, white2binary

class WSIdataset(data.Dataset):
    def __init__(self, lib='', transform=None, shuffle=False):
        # opens data dictionary (lib)
        tar = lib['targets']
        all_slides = lib['paths']
        
        tum_score = []
        tiles = []
        ntiles = []
        slideIDX = []
        targets = []
        slides = []
        j = 0
        for i, path in enumerate(all_slides):
            t = []
            # g = []
            if os.path.exists(path):
                for f in os.listdir(path):
                    tum_p = float(f[:-4].split('-')[-1])
                    if (f[-4:]=='.jpg' or f[-4:]=='.png'): #and tum_p >= 0.8:
                        t.append(os.path.join(path, f)) #paths to all tiles of the slide
                if len(t) >= 10: #processing if slide gets >= than 10 tiles 
                    slides.append(path)          # slide path
                    tiles.extend(t)              # all tiles of the data set
                    ntiles.append(len(t))        # number of tiles in this slide
                    slideIDX.extend([j]*len(t))  # array of index --- all tiles of same slide gets unique index
                    targets.append(tar[i])       # slide groudn truth label as the tile ground truth label
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
        self.transform = transform
        self.mode = None
        self.shuffle = shuffle

    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs): # making data set using selected tiles
        self.t_data = [(self.slideIDX[x],self.tiles[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1: # mode to laod all tiles of the given data set (used to laod valdiation and test set)
            tile = self.tiles[index]
            img = Image.open(str(tile)).convert('RGB')            
            slideIDX = self.slideIDX[index]
            target = self.targets[slideIDX]
            img = img.resize((256,256),Image.BILINEAR)                
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        elif self.mode == 2: #mode to load selected tiles of the given data set (used to load IDaRS train set)
            slideIDX, tile, target = self.t_data[index]
            img = Image.open(str(tile)).convert('RGB')
            img = img.resize((256,256),Image.BILINEAR)                
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.tiles)
        elif self.mode == 2:
            return len(self.t_data)

class featuresdataset(data.Dataset):
    def __init__(self, data_path, data_frame, raw_images=False, transform=None, stain_aug=False, multi=None):
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
        vendors = []
        j = 0
        for i, case in enumerate(list_of_images):
            cohort = list_of_cohorts[i]
            label = data_frame.loc[(data_frame['wsi']==case) & (data_frame['cohort']==cohort)]['label'].item()
            vendor = data_frame.loc[(data_frame['wsi']==case) & (data_frame['cohort']==cohort)]['vendor'].item()
            t = []
            if type(self.data_path) == list:
                if int(cohort) == 3:
                    path = os.path.join(self.data_path[0], case)
                elif int(cohort) == 4:
                    path = os.path.join(self.data_path[1], case)
                elif int(cohort) == 1:
                    path = os.path.join(self.data_path[2], case)   
            else:
                path = os.path.join(self.data_path, case)

            # if os.path.exists(path):
            for f in os.listdir(path):
                t.append(os.path.join(path, f)) #paths to all tiles of the slide
            # if len(t) >= 10: #processing if slide gets >= than 10 tiles 
            slides.append(path)          # slide path
            tiles.extend(t)              # all tiles of the data set
            ntiles.append(len(t))        # number of tiles in this slide
            slideIDX.extend([j]*len(t))  # array of index --- all tiles of same slide gets unique index
            targets.append(label)        # slide groudn truth label as the tile ground truth label
            cohorts.append(cohort)       # some slides are repated in cohorts, this locates the correct slide
            vendors.append(vendor)
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
        self.vendors = vendors
        self.stain_aug = stain_aug
        self.multi = multi

        if stain_aug == True:
            stain_matrix = np.array([[0.91633014, -0.20408072, -0.34451435],
                [0.17669817, 0.92528011, 0.33561059]])
            self.stain_augmentor = StainAugmentor(
                "macenko",
                stain_matrix=stain_matrix,
                sigma1=0.2,
                sigma2=0.2,
                augment_background=True)

    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs): # making data set using selected tiles
        self.t_data = [(self.slideIDX[x],self.tiles[x],self.targets[self.slideIDX[x]],self.cohorts[self.slideIDX[x]], self.vendors[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def reduce_ftrs(self, features):
        import pandas as pd
        blank_graph_df = pd.read_csv("/home/adams/Projects/ANTICIPATE/outcome_prediction/feature_csvs/graph_2class_reduced.csv", index_col=0, names=['values'])
        blank_morph_df = pd.read_csv("/home/adams/Projects/ANTICIPATE/outcome_prediction/feature_csvs/morph_2class_reduced.csv", index_col=0, names=['values'])
        ftr_list = [*blank_morph_df.index.tolist(), *blank_graph_df.index.tolist()]
        discard = ["Kurtosis", "Skewness", "kurtosis", "skew", "Voronoi", "Delaunay"]
        ftrs = features#.numpy()
        ftr_df = pd.DataFrame([ftrs], columns=ftr_list)
        for name in discard:
            ftr_df = ftr_df[ftr_df.columns.drop(list(ftr_df.filter(regex=name)))]
        ftrs_edited = ftr_df.to_numpy()[0,:] #[0,:104]
        features = torch.tensor(ftrs_edited)
        return features
    def add_to_stain(self, img):
        img = np.asarray(img)
        """Perturbe the stain intensities of input images, i.e. stain augmentation."""
        ret = np.uint8(self.stain_augmentor.apply(np.uint8(img.copy())))
        return ret

    def __getitem__(self,index):
        if self.mode == 1: # mode to laod all tiles of the given data set (used to laod valdiation and test set)
            tile = self.tiles[index]
            case = os.path.basename(tile).split('_')[0]
            slideIDX = self.slideIDX[index]
            target = self.targets[slideIDX]
            vendor = self.vendors[slideIDX]
            if self.raw_images is True:
                img = Image.open(str(tile)).convert('RGB')
                # if self.stain_aug == True:
                #     img = self.add_to_stain(img)
                img = img.resize((256,256),Image.BILINEAR)  
                if self.transform is not None:
                    img = self.transform(img)
            else:
                img = torch.load(str(tile))
                img = self.reduce_ftrs(img)
                if self.multi is not None:
                    img_tile = tile.replace(self.multi[0], self.multi[1])# swap for image path...
                    img_tile = img_tile.replace('.pt','.png')
                    img2 = Image.open(str(img_tile)).convert('RGB')
                    img2 = img2.resize((256,256),Image.BILINEAR)  
                    if self.transform is not None:
                        img2 = self.transform(img2)
                    return [(img, img2), (target, vendor, case)]              
            return [img, (target, vendor, case)]
        elif self.mode == 2: #mode to load selected tiles of the given data set (used to load IDaRS train set)
            slideIDX, tile, target, coh, vendor = self.t_data[index]
            case = os.path.basename(tile).split('_')[0]
            if self.raw_images is True:
                img = Image.open(str(tile)).convert('RGB')
                # if self.stain_aug == True:
                #     img = self.add_to_stain(img)
                # img = Image.fromarray(img).resize((256,256),Image.BILINEAR) 
                img = img.resize((256,256),Image.BILINEAR)
                if self.transform is not None:
                    img = self.transform(img)
            else:
                img = torch.load(str(tile))
                img = self.reduce_ftrs(img)
                if self.multi is not None:
                    img_tile = tile.replace(self.multi[0], self.multi[1])# swap for image path...
                    img_tile = img_tile.replace('.pt','.png')
                    img2 = Image.open(str(img_tile)).convert('RGB')
                    img2 = img2.resize((256,256),Image.BILINEAR)  
                    if self.transform is not None:
                        img2 = self.transform(img2)
                    return [(img, img2), (target, vendor, case)]              
            return [img, (target, vendor, case)]

    def __len__(self):
        if self.mode == 1:
            return len(self.tiles)
        elif self.mode == 2:
            return len(self.t_data)


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
            if type(self.data_path) == list:
                if int(cohort) == 3:
                    path = os.path.join(self.data_path[0], case)
                elif int(cohort) == 4:
                    path = os.path.join(self.data_path[1], case)
                elif int(cohort) == 1:
                    path = os.path.join(self.data_path[2], case)   
            else:
                path = os.path.join(self.data_path, case)

            # if os.path.exists(path):
            for f in os.listdir(path):
                t.append(os.path.join(path, f)) #paths to all tiles of the slide
            # if len(t) >= 10: #processing if slide gets >= than 10 tiles 
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

    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs): # making data set using selected tiles
        self.t_data = [(self.slideIDX[x],self.tiles[x],self.targets[self.slideIDX[x]],self.cohorts[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def reduce_ftrs(self, features):
        import pandas as pd
        blank_graph_df = pd.read_csv("/home/adams/Projects/ANTICIPATE/outcome_prediction/feature_csvs/graph_2class_reduced.csv", index_col=0, names=['values'])
        blank_morph_df = pd.read_csv("/home/adams/Projects/ANTICIPATE/outcome_prediction/feature_csvs/morph_2class_reduced.csv", index_col=0, names=['values'])
        ftr_list = [*blank_morph_df.index.tolist(), *blank_graph_df.index.tolist()]
        discard = ["Kurtosis", "Skewness", "kurtosis", "skew", "Voronoi", "Delaunay"]
        ftrs = features#.numpy()
        ftr_df = pd.DataFrame([ftrs], columns=ftr_list)
        for name in discard:
            ftr_df = ftr_df[ftr_df.columns.drop(list(ftr_df.filter(regex=name)))]
        ftrs_edited = ftr_df.to_numpy()[0,:] #[0,:104]
        features = torch.tensor(ftrs_edited)
        return features

    def __getitem__(self,index):
        if self.mode == 1: # mode to laod all tiles of the given data set (used to laod valdiation and test set)
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
                img = self.reduce_ftrs(img)      
            return [img, (target, case)]
        elif self.mode == 2: #mode to load selected tiles of the given data set (used to load IDaRS train set)
            slideIDX, tile, target, coh = self.t_data[index]
            case = os.path.basename(tile).split('_')[0]
            if self.raw_images is True:
                img = Image.open(str(tile)).convert('RGB')
                img = img.resize((256,256),Image.BILINEAR)
                if self.transform is not None:
                    img = self.transform(img)
            else:
                img = torch.load(str(tile))
                img = self.reduce_ftrs(img)          
            return [img, (target, vendor, case)]

    def __len__(self):
        if self.mode == 1:
            return len(self.tiles)
        elif self.mode == 2:
            return len(self.t_data)

class featuresdataset_wsi_inference(data.Dataset):
    def __init__(self, wsi_path, mask_path, layer_col_dict, layer_res, patch_size, stride, out_res, epith_thresh, raw_images=False, transform=None):
        # opens data dictionary (lib)
        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.color_dict = layer_col_dict
        patches, rois = self.create_patches(wsi_path, mask_path, layer_col_dict, layer_res, patch_size, stride, out_res, epith_thresh)
        self.patches = patches
        self.rois = rois
        self.transform = transform
        self.raw_images = raw_images

    def create_patches(self, wsi_path, mask_path, layer_col_dict, layer_res=0.5, patch_size=512, stride=256, out_res=0.5, epith_thresh=0.5):
        wsi = WSIReader.open(wsi_path)
        mask = imread(mask_path)

        meta = WSIMeta(
            slide_dimensions=tuple(wsi.slide_dimensions(layer_res, 'mpp')), level_downsamples=[1, 2, 4, 8],
            mpp=tuple(wsi.convert_resolution_units(layer_res, 'mpp')['mpp']), objective_power=5, axes='YXS'
        )

        layers_new = mask2epithmask(mask, layer_col_dict, labels=[2,3,4])
        layer_mask = VirtualWSIReader(layers_new, info=meta)

        # if any(item in [2,3,4] for item in np.unique(layers)) == False:
        if np.max(layers_new) == 0: # i.e. no epith detected
            return

        patches = []
        patch_rois = []
        img_patches = patchextraction.get_patch_extractor(
            input_img=wsi,
            input_mask=layer_mask,
            method_name="slidingwindow",
            patch_size=patch_size,
            resolution=out_res,
            units="mpp", #"power",
            stride=stride,
            pad_mode="constant",
            pad_constant_values=255,
            within_bound=False,
        )

        for patch in img_patches:
            # 1. Get layer mask
            item = img_patches.n-1
            bounds = img_patches.coordinate_list[item]
            patch_mask = layer_mask.read_bounds(
                        bounds,
                        resolution=out_res,
                        units="mpp", #"power",
                        interpolation="nearest",
                        pad_mode="constant",
                        pad_constant_values=0,
                        coord_space="resolution",
            )
            # patch_mask_g = colour2gray(patch_mask, layer_col_dict)
            # REMEMBER EPITH NUCLEI ARE BEING DEFAULTED TO NOT BE IN KERATIN/BASAL LAYER
            # patch_mask_binary = np.where(patch_mask_g > 0, 1, 0)
            # epith_mask_binary = np.where(patch_mask_g > 1, 1, 0)
            epith_mask_binary = white2binary(patch_mask)

            if np.mean(epith_mask_binary) < epith_thresh:
                continue

            patches.append(patch)
            patch_rois.append(bounds)
        
        return patches, patch_rois

    def reduce_ftrs(self, features):
        import pandas as pd
        blank_graph_df = pd.read_csv("/home/adams/Projects/ANTICIPATE/outcome_prediction/feature_csvs/graph_2class_reduced.csv", index_col=0, names=['values'])
        blank_morph_df = pd.read_csv("/home/adams/Projects/ANTICIPATE/outcome_prediction/feature_csvs/morph_2class_reduced.csv", index_col=0, names=['values'])
        ftr_list = [*blank_morph_df.index.tolist(), *blank_graph_df.index.tolist()]
        discard = ["Kurtosis", "Skewness", "kurtosis", "skew", "Voronoi", "Delaunay"]
        ftrs = features#.numpy()
        ftr_df = pd.DataFrame([ftrs], columns=ftr_list)
        for name in discard:
            ftr_df = ftr_df[ftr_df.columns.drop(list(ftr_df.filter(regex=name)))]
        ftrs_edited = ftr_df.to_numpy()[0,:] #[0,:104]
        features = torch.tensor(ftrs_edited)
        return features

    def __getitem__(self,index):
        tile = self.patches[index]
        coords = self.rois[index]
        if self.raw_images is True:
            img = Image.fromarray(tile)
            img = img.resize((256,256),Image.BILINEAR)  
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = torch.load(str(tile))
            img = self.reduce_ftrs(img)          
        return [img, coords]

    def __len__(self):
        return len(self.patches)

class featuresdataset_wsi_mlp_inference(data.Dataset):
    def __init__(self, patches, rois, layer_col_dict):
        self.color_dict = layer_col_dict
        self.patches = patches
        self.rois = rois

    def reduce_ftrs(self, features):
        import pandas as pd
        blank_graph_df = pd.read_csv("/home/adams/Projects/ANTICIPATE/outcome_prediction/feature_csvs/graph_2class_reduced.csv", index_col=0, names=['values'])
        blank_morph_df = pd.read_csv("/home/adams/Projects/ANTICIPATE/outcome_prediction/feature_csvs/morph_2class_reduced.csv", index_col=0, names=['values'])
        ftr_list = [*blank_morph_df.index.tolist(), *blank_graph_df.index.tolist()]
        discard = ["Kurtosis", "Skewness", "kurtosis", "skew", "Voronoi", "Delaunay"]
        ftrs = features#.numpy()
        ftr_df = pd.DataFrame([ftrs], columns=ftr_list)
        for name in discard:
            ftr_df = ftr_df[ftr_df.columns.drop(list(ftr_df.filter(regex=name)))]
        ftrs_edited = ftr_df.to_numpy()[0,:] #[0,:104]
        ftrs_edited = ftrs_edited.astype(np.float)
        features = torch.from_numpy(ftrs_edited)
        return features

    def __getitem__(self,index):
        tile = self.patches[index]
        coords = torch.from_numpy(self.rois[index])
        # img = torch.load(str(tile))
        img = torch.from_numpy(tile)
        img = self.reduce_ftrs(img)          
        return [img, coords]

    def __len__(self):
        return len(self.patches)
