
from collections import OrderedDict
import numpy as np
import cv2


def get_inst_stat(inst_id, inst_dict, slide, nuc_type):
    inst_cnt = np.array(inst_dict['contour']).astype(np.float32)
    inst_box = np.array([
        [np.min(inst_cnt[:,1]), np.min(inst_cnt[:,0])],
        [np.max(inst_cnt[:,1]), np.max(inst_cnt[:,0])]
    ])
    bbox_h, bbox_w = inst_box[1] - inst_box[0]
    bbox_aspect_ratio = float(bbox_w / bbox_h)
    bbox_area = bbox_h * bbox_w
    contour_area = cv2.contourArea(inst_cnt)
    extent = float(contour_area) / bbox_area
    convex_hull = cv2.convexHull(inst_cnt)
    convex_area = cv2.contourArea(convex_hull)
    convex_area = convex_area if convex_area != 0 else 1
    solidity = float(contour_area) / convex_area
    equiv_diameter = np.sqrt(4 * contour_area / np.pi)
    if inst_cnt.shape[0] > 4:
        _, axes, orientation = cv2.fitEllipse(inst_cnt)
        major_axis_length = max(axes)
        minor_axis_length = min(axes)
    else:
        orientation = 0
        major_axis_length = 1
        minor_axis_length = 1
    perimeter = cv2.arcLength(inst_cnt, True)
    _, radius = cv2.minEnclosingCircle(inst_cnt)
    eccentricity = np.sqrt(1- (minor_axis_length / major_axis_length)**2)
    stat_dict = OrderedDict()
    stat_dict['nuc_id'] = inst_id
    stat_dict['wsi_id'] = slide
    stat_dict['type'] = nuc_type #inst_dict['type']
    stat_dict['eccentricity'] = eccentricity
    stat_dict['convex_area'] = convex_area
    stat_dict['contour_area'] = contour_area
    stat_dict['equiv_diameter'] = equiv_diameter
    stat_dict['extent'] = extent
    stat_dict['major_axis_length'] = major_axis_length
    stat_dict['minor_axis_length'] = minor_axis_length
    stat_dict['perimeter'] = perimeter
    stat_dict['solidity'] = solidity
    stat_dict['orientation'] = orientation
    stat_dict['radius'] = radius
    stat_dict['bbox_area'] = bbox_area
    stat_dict['bbox_aspect_ratio'] = bbox_aspect_ratio
    return inst_id, stat_dict