from eolearn.core import EOPatch, FeatureType, OverwritePermission
import numpy as np
from sklearn.utils import resample
import random
import pandas as pd
import collections
import time
import datetime as dt
import random
from temporal_features import AddStreamTemporalFeaturesTask as st
from all_stream_features import allValid
import itertools as it
from os import path as ospath
from eolearn.core import EOTask, FeatureType
from scipy.ndimage import gaussian_filter
from eolearn.ml_tools.utilities import rolling_window
from os import path as ospath

from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadFromDisk, SaveTask, EOExecutor
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from eolearn.io import S2L1CWCSInput, S2L1CWMSInput
from shapely.geometry import Polygon
import os
from sentinelhub import BBoxSplitter, BBox, CRS, CustomUrlParam
import datetime as dt
import time


def s1_iw_ew():
    patches_path_s1 = 'E:\\Data\\PerceptiveSentinel\\Slovenia_S1'

    for patch_id in range(1061):
        p2 = '{}/eopatch_{}'.format(patches_path_s1, int(patch_id))
        if not ospath.exists(p2):
            print('Patch {} is missing.'.format(patch_id))
        else:
            eopatch_s1 = EOPatch.load(p2, lazy_loading=True)
            print('{} - {}'.format(patch_id, str(eopatch_s1.data['IW'].shape)))


def bbox_mystery_case():
    DATA_FOLDER = os.path.join('data')

    area = gpd.read_file(os.path.join(DATA_FOLDER, 'svn_buffered.geojson'))
    # area = gpd.read_file(os.path.join(DATA_FOLDER, 'austria.geojson'))

    # Convert CRS to UTM_33N
    country_crs = CRS.UTM_33N
    area = area.to_crs(crs={'init': CRS.ogc_string(country_crs)})

    # Get the country's shape in polygon format
    country_shape = area.geometry.values.tolist()[-1]
    # Create the splitter to obtain a list of bboxes
    bbox_splitter = BBoxSplitter([country_shape], country_crs, (25 * 2, 17 * 2))
    bbox_list = np.array(bbox_splitter.get_bbox_list())

    path = 'E:\\Data\\PerceptiveSentinel\\Slovenia'
    for i, bb in enumerate(bbox_list):
        if ospath.exists('{}\\eopatch_{}'.format(path, i)):
            eopatch = EOPatch.load('{}/eopatch_{}'.format(path, i), lazy_loading=True)
            if bb == eopatch.bbox:
                print('{} match'.format(i))
            else:
                print('{} XX MISMATCH XX'.format(i))
        else:
            print('{} -- missing'.format(i))


def missing_cloud_mask():
    DATA_FOLDER = os.path.join('data')

    area = gpd.read_file(os.path.join(DATA_FOLDER, 'svn_buffered.geojson'))
    # area = gpd.read_file(os.path.join(DATA_FOLDER, 'austria.geojson'))

    # Convert CRS to UTM_33N
    country_crs = CRS.UTM_33N
    area = area.to_crs(crs={'init': CRS.ogc_string(country_crs)})

    # Get the country's shape in polygon format
    country_shape = area.geometry.values.tolist()[-1]
    # Create the splitter to obtain a list of bboxes
    bbox_splitter = BBoxSplitter([country_shape], country_crs, (25 * 2, 17 * 2))
    bbox_list = np.array(bbox_splitter.get_bbox_list())

    path = 'E:\\Data\\PerceptiveSentinel\\Slovenia'
    for i, bb in enumerate(bbox_list):
        if not ospath.exists('{}\\eopatch_{}\\mask\\CLM.npy'.format(path, i)):
            print(i)


def check_dem():
    patches_path_s1 = 'E:\\Data\\PerceptiveSentinel\\Slovenia'

    for patch_id in range(1084):
        p2 = '{}/eopatch_{}'.format(patches_path_s1, int(patch_id))
        if not ospath.exists(p2):
            print('Patch {} is missing.'.format(patch_id))
        else:
            try:
                eopatch_s1 = EOPatch.load(p2, lazy_loading=True)
                sh = eopatch_s1.data_timeless['INCLINATION'].shape
                if sh[0] != 505:
                    print('{} {}'.format(patch_id, sh))
            except:
                print('{} missing'.format(patch_id))


def count_features():
    fet = 'DEM,ARVI_diff_diff,ARVI_diff_max,ARVI_diff_min,ARVI_max_mean_feature,ARVI_max_mean_len,ARVI_max_mean_surf,ARVI_max_val,ARVI_mean_val,ARVI_min_val,ARVI_neg_len,ARVI_neg_rate,ARVI_neg_surf,ARVI_neg_tran,ARVI_pos_len,ARVI_pos_rate,ARVI_pos_surf,ARVI_pos_tran,ARVI_sd_val,EVI_diff_diff,EVI_diff_max,EVI_diff_min,EVI_max_mean_feature,EVI_max_mean_len,EVI_max_mean_surf,EVI_max_val,EVI_mean_val,EVI_min_val,EVI_neg_len,EVI_neg_rate,EVI_neg_surf,EVI_neg_tran,EVI_pos_len,EVI_pos_rate,EVI_pos_surf,EVI_pos_tran,EVI_sd_val,NDVI_diff_diff,NDVI_diff_max,NDVI_diff_min,NDVI_max_mean_feature,NDVI_max_mean_len,NDVI_max_mean_surf,NDVI_max_val,NDVI_mean_val,NDVI_min_val,NDVI_neg_len,NDVI_neg_rate,NDVI_neg_surf,NDVI_neg_tran,NDVI_pos_len,NDVI_pos_rate,NDVI_pos_surf,NDVI_pos_tran,NDVI_sd_val,NDWI_diff_diff,NDWI_diff_max,NDWI_diff_min,NDWI_max_mean_feature,NDWI_max_mean_len,NDWI_max_mean_surf,NDWI_max_val,NDWI_mean_val,NDWI_min_val,NDWI_neg_len,NDWI_neg_rate,NDWI_neg_surf,NDWI_neg_tran,NDWI_pos_len,NDWI_pos_rate,NDWI_pos_surf,NDWI_pos_tran,NDWI_sd_val,SIPI_diff_diff,SIPI_diff_max,SIPI_diff_min,SIPI_max_mean_feature,SIPI_max_mean_len,SIPI_max_mean_surf,SIPI_max_val,SIPI_mean_val,SIPI_min_val,SIPI_neg_len,SIPI_neg_rate,SIPI_neg_surf,SIPI_neg_tran,SIPI_pos_len,SIPI_pos_rate,SIPI_pos_surf,SIPI_pos_tran,SIPI_sd_val,SAVI_diff_diff,SAVI_diff_max,SAVI_diff_min,SAVI_max_mean_feature,SAVI_max_mean_len,SAVI_max_mean_surf,SAVI_max_val,SAVI_mean_val,SAVI_min_val,SAVI_neg_len,SAVI_neg_rate,SAVI_neg_surf,SAVI_neg_tran,SAVI_pos_len,SAVI_pos_rate,SAVI_pos_surf,SAVI_pos_tran,SAVI_sd_val,BLUE_diff_diff,BLUE_diff_max,BLUE_diff_min,BLUE_max_mean_feature,BLUE_max_mean_len,BLUE_max_mean_surf,BLUE_max_val,BLUE_mean_val,BLUE_min_val,BLUE_neg_len,BLUE_neg_rate,BLUE_neg_surf,BLUE_neg_tran,BLUE_pos_len,BLUE_pos_rate,BLUE_pos_surf,BLUE_pos_tran,BLUE_sd_val,GREEN_diff_diff,GREEN_diff_max,GREEN_diff_min,GREEN_max_mean_feature,GREEN_max_mean_len,GREEN_max_mean_surf,GREEN_max_val,GREEN_mean_val,GREEN_min_val,GREEN_neg_len,GREEN_neg_rate,GREEN_neg_surf,GREEN_neg_tran,GREEN_pos_len,GREEN_pos_rate,GREEN_pos_surf,GREEN_pos_tran,GREEN_sd_val,RED_diff_diff,RED_diff_max,RED_diff_min,RED_max_mean_feature,RED_max_mean_len,RED_max_mean_surf,RED_max_val,RED_mean_val,RED_min_val,RED_neg_len,RED_neg_rate,RED_neg_surf,RED_neg_tran,RED_pos_len,RED_pos_rate,RED_pos_surf,RED_pos_tran,RED_sd_val,NIR_diff_diff,NIR_diff_max,NIR_diff_min,NIR_max_mean_feature,NIR_max_mean_len,NIR_max_mean_surf,NIR_max_val,NIR_mean_val,NIR_min_val,NIR_neg_len,NIR_neg_rate,NIR_neg_surf,NIR_neg_tran,NIR_pos_len,NIR_pos_rate,NIR_pos_surf,NIR_pos_tran,NIR_sd_val,VV_avg,VV_max,VV_min,VV_std,VH_avg,VH_max,VH_min,VH_std,VV_spring_avg,VV_spring_max,VV_spring_min,VV_spring_std,VV_summer_avg,VV_summer_max,VV_summer_min,VV_summer_std,VV_autumn_avg,VV_autumn_max,VV_autumn_min,VV_autumn_std,VV_winter_avg,VV_winter_max,VV_winter_min,VV_winter_std,VH_spring_avg,VH_spring_max,VH_spring_min,VH_spring_std,VH_summer_avg,VH_summer_max,VH_summer_min,VH_summer_std,VH_autumn_avg,VH_autumn_max,VH_autumn_min,VH_autumn_std,VH_winter_avg,VH_winter_max,VH_winter_min,VH_winter_std,'
    c = 0
    for f in fet:
        if f == ',':
            c += 1
    print(c)


if __name__ == '__main__':
    # bbox_mystery_case()
    # missing_cloud_mask()
    # check_dem()
    count_features()