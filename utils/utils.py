import numpy as np
import cv2
from tiatoolbox.wsicore.wsireader import WSIReader
import h5py

def mask2epithmask(img, color_dict, labels=[2,3,4]):
    for l in color_dict:
        # if l < 2:
        if l not in labels:
            img[np.where(np.all(img==color_dict[l], axis=2))] = (0,0,0)
        else:
            img[np.where(np.all(img==color_dict[l], axis=2))] = (255,255,255)
    return img

def white2binary(img):
    img_new = np.zeros_like(img[...,0])
    img_new[np.where(np.all(img==(255,255,255), axis=2))] = 1
    return img_new

def colour2gray(img, color_dict):
    img_new = np.zeros_like(img[...,0])
    for l in color_dict:
        img_new[np.where(np.all(img==color_dict[l], axis=2))] = l
    img_new[np.where(np.all(img==(255,255,255), axis=2))] = 3
    return img_new

def get_heatmap(image):
    image = ((image/np.max(image))*255).astype('uint8')
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap[(heatmap[:,:,0]==0) & (heatmap[:,:,1]==0) & (heatmap[:,:,2]==128)] = [0,0,0]
    return heatmap

def build_heatmap(wsi_path, out_res, proc_res, coords, probs):
    wsi = WSIReader.open(wsi_path)
    proc_dims = wsi.slide_dimensions(resolution=proc_res, units='mpp')
    out_dims = wsi.slide_dimensions(resolution=out_res, units='mpp')
    ds = out_dims/proc_dims
    map = np.zeros(out_dims[::-1])
    counts = np.zeros(out_dims[::-1])
    for idx, prob in enumerate(probs):
        x1, y1, x2, y2 = coords[idx]
        x1, x2 = int(x1*ds[0]), int(x2*ds[0])
        y1, y2 = int(y1*ds[1]), int(y2*ds[1])
        y2 = min(y2, map.shape[0])
        x2 = min(x2, map.shape[1])
        map[y1:y2, x1:x2] += np.ones((y2-y1, x2-x1))*prob
        counts[y1:y2, x1:x2] += np.ones((y2-y1, x2-x1))
    heatmap = map / counts
    heatmap[np.isnan(heatmap)] = 0
    return heatmap

def decolourise(image, col_dict):
    image_col = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    for _, col in col_dict.items():
        image_col[np.where(np.all(image == col[1], axis=-1))] = col[0]
    return image_col

def colourise(image, col_dict):
    image_col = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')
    for _, col in col_dict.items():
        try:
            image_col[image==col[0]] = col[1]
        except:
            continue
    return image_col

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path