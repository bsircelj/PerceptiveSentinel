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
from all_stream_features import AddBaseFeatures, allValid

if __name__ == '__main__':
    samples_path = '/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/enriched_samples9797.csv'  # CHANGE

    # patches_path = 'E:/Data/PerceptiveSentinel/Slovenia'
    patches_path = '/home/beno/Documents/test/Slovenia'

    dataset = pd.read_csv(samples_path)
    no = dataset.shape[0]

    # red = np.zeros(no)
    # green = np.zeros(no)
    # blue = np.zeros(no)
    bands = np.zeros((69, 1, no, 13))

    for x in range(1):  # CHANGE
        patch_id = dataset['patch_no'][x]
        w = dataset['x'][x]
        h = dataset['y'][x]
        w = np.clip(w, 0, 300)
        h = np.clip(h, 0, 300)

        eopatch = EOPatch.load('{}/eopatch_{}'.format(patches_path, 0), lazy_loading=True)
        for i in range(13):
            for time in range(69):
                bands[time, 0, x, i] = eopatch.data['BANDS'][time][h][w][i]

    new_patch = EOPatch()
    new_patch.add_feature(FeatureType.DATA, 'BANDS', bands)
    for x in range(13):
        one_band = bands[..., x]
        one_band = np.expand_dims(one_band, axis=-1)
        new_patch.add_feature(FeatureType.DATA, 'B{}'.format(int(x + 1)), one_band)

    new_patch.timestamp = EOPatch.load('{}/eopatch_{}'.format(patches_path, 0), lazy_loading=True).timestamp  # CHANGE
    # new_patch.add_feature(FeatureType.MASK, 'VALID_DATA', np.ones((69, 1, no, 1)))
    new_patch = allValid('VALID_DATA').execute(new_patch)

    new_patch = AddBaseFeatures().execute(new_patch)
    features = ['NDVI', 'SAVI', 'SIPI', 'EVI', 'ARVI', 'NDWI']
    for f in features:
        print(f + '\n')
        new_patch = st(data_feature=f).execute(new_patch)

    print('BLUE\n')
    new_patch = st(data_feature='B2', feature_name_prefix='BLUE', ).execute(new_patch)
    print('GREEN\n')
    new_patch = st(data_feature='B3', feature_name_prefix='GREEN').execute(new_patch)
    print('RED\n')
    new_patch = st(data_feature='B4', feature_name_prefix='RED').execute(new_patch)
    print('NIR\n')
    new_patch = st(data_feature='B8', feature_name_prefix='NIR').execute(new_patch)

    print(new_patch)
    all_features = new_patch.get_feature_list()
    all_data_timeless = []
    for f in all_features:
        if type(f) is tuple:
            all_data_timeless.append((f[0], f[1]))

    for f in all_data_timeless:
        dataset[f] = new_patch.data_timeless[f].squeeze()

    filename = 'extended_samples' + str(int(random.random() * 10000))

    dataset.to_csv('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/' + filename + '.csv')
