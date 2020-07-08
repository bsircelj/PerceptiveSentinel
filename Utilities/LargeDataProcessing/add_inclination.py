from eolearn.core import EOPatch, FeatureType, OverwritePermission
import numpy as np
from sklearn.utils import resample
import random
import pandas as pd
import collections
import time
import datetime as dt
import random
import itertools as it
from os import path as ospath
from eolearn.core import EOTask, FeatureType
from scipy.ndimage import gaussian_filter
from eolearn.ml_tools.utilities import rolling_window

if __name__ == '__main__':
    # samples_path = '/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/enriched_samples9797.csv'  # CHANGE

    print('start ' + str(dt.datetime.now()))
    # samples_path = '/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/'  # CHANGE
    samples_path = 'D:\\Samples\\'
    patches_path = 'E:/Data/PerceptiveSentinel/SVN/2017/processed/patches/'
    # patches_path = '/home/beno/Documents/test/Slovenia'

    dataset = pd.read_csv(samples_path + 'final_g2_1.csv')
    # dataset.drop(columns=['Unnamed: 0', 'NDVI_sd_val', 'EVI_min_val', 'ARVI_max_mean_len', 'SIPI_mean_val',
    #                       'NDVI_min_val', 'SAVI_min_val'], inplace=True)
    dataset.sort_values(by='patch_no', inplace=True)
    no = dataset.shape[0]

    dataset['INCLINATION'] = np.zeros(no)
    patch_id = -1
    eopatch = EOPatch()
    for x in range(no):
        patch_id_new = dataset['patch_no'][x]
        w = int(dataset['x'][x])
        h = int(dataset['y'][x])

        if patch_id != patch_id_new:

            p1 = '{}/eopatch_{}'.format(patches_path, int(patch_id_new))
            if not ospath.exists(p1):
                print('Patch {} is missing.'.format(patch_id_new))
                continue
            patch_id = patch_id_new
            print(patch_id)
            eopatch = EOPatch.load(p1, lazy_loading=True)  # CHANGE

        dataset['INCLINATION'][x] = eopatch.data_timeless['INCLINATION'][h][w].squeeze()

    filename = ''
    i = 0
    while True:
        filename = 'final_g2_' + str(i)
        if ospath.exists(samples_path + filename + '.csv'):
            i += 1
        else:
            break

    # print(dataset)
    dataset = pd.DataFrame.from_dict(dataset)
    # print(dataset)
    dataset.to_csv(samples_path + filename + '.csv', index=False)
