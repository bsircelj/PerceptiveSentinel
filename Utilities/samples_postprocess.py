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


def save_csv(data, path, filename):
    i = 0
    while True:
        filename_temp = filename + str(i)
        if ospath.exists(samples_path + filename_temp + '.csv'):
            i += 1
        else:
            break
    data.to_csv(path + filename_temp + '.csv', index=False)


if __name__ == '__main__':
    samples_path = 'D:/Samples/'

    df = pd.read_csv(samples_path + 'g2_extended1.csv')
    df.drop(inplace=True, columns=['Unnamed: 0'])
    base_names = ['ARVI', 'EVI', 'NDVI', 'NDWI', 'SIPI', 'SAVI', 'BLUE', 'GREEN', 'RED', 'NIR']

    indexes_to_drop = []
    for index, row in df.iterrows():
        if index % 10000 == 0:
            print(index/10000)
        for b in base_names:
            if row[b + '_max_val'] == 0:
                indexes_to_drop.append(index)
    df.drop(inplace=True, index=indexes_to_drop)
    # Change to new class mapping
    class_dictionary = collections.Counter(df['LPIS_2017_G2'])
    class_count = class_dictionary.most_common()
    print(class_count)
    least_common = class_count[-1][1]
    occurrences = np.zeros(15)
    indexes_to_drop = []
    for index, row in df.iterrows():
        if index % 10000 == 0:
            print('removal ' + str(index/10000))
        cls = int(row['LPIS_2017_G2'])
        occurrences[cls] += 1
        if occurrences[cls] > least_common:
            indexes_to_drop.append(index)
    print('least common: ' + str(least_common))
    print('class size ' + str(df['LPIS_2017_G2'].shape))
    df.drop(inplace=True, index=indexes_to_drop)
    save_csv(df, samples_path, 'final_g2_')
