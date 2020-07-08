"""Prepare subset of the data for modelling and evaluation.
"""
import collections
import os
import random
import re
import sys
import time
# from dotenv import find_dotenv, load_dotenv
from pathlib import Path

import click
import numpy as np
import pandas as pd
from eolearn.core import EOExecutor, EOPatch, FeatureType, LinearWorkflow, \
    LoadTask, OverwritePermission, SaveTask
from eolearn.geometry import PointSamplingTask
from sklearn.utils import resample
import os.path as ospath
import datetime as dt

# Global variables
log = None
report = {}


def sample_patches(
        path,
        patches,
        no_samples,
        class_feature,
        mask_feature,
        features,
        weak_classes,
        samples_per_class=None,
        debug=False,
        seed=None,
        class_frequency=False
):
    """
    :param path: Path to folder containing all patches, folders need to be
        named eopatch_{number: 0 to no_patches-1}
    :type path: string
    :param patches: List of patch IDs, e.g. [0, 1, ...]
    :type patches: list[int]
    :param no_samples: Number of samples taken per patch
    :type no_samples: int
    :param class_feature: Name of feature that contains class number.
        The numbers in the array can be float or nan for no class
    :type class_feature: (FeatureType, string)
    :param mask_feature: Feature that defines the area from where samples are
        taken, if None the whole image is used
    :type mask_feature: (FeatureType, String) or None
    :param features: Features to include in returned dataset for each pixel
        sampled
    :type features: array of type [(FeatureType, string), ...]
    :param samples_per_class: Number of samples per class returned after
        balancing. If the number is higher than minimal number of samples for
        the smallest class then those numbers are upsampled by repetition.
        If the argument is None then number is set to the size of the number of
        samples of the smallest class
    :type samples_per_class: int or None
    :param debug: If set to True patch id and coordinates are included in
        returned DataFrame
    :param seed: Seed for random generator
    :return: pandas DataFrame with columns
        [class feature, features, patch_id, x coord, y coord].
        id,x and y are used for testing
    :param class_frequency: If set to True, the function also return
        dictionary of each class frequency before balancing
    :type class_frequency: boolean
    :param weak_classes: Classes that when found also the neighbouring regions
        will be checked and added if they contain one of the weak classes.
        Used to enrich the samples
    :type weak_classes: int list
    """
    if seed is not None:
        random.seed(seed)

    columns = [class_feature[1]] + [x[1] for x in features]
    if debug:
        columns = columns + ['patch_no', 'x', 'y']

    class_name = class_feature[1]
    sample_dict = []

    for patch_id in patches:
        print(patch_id)
        eopatch = EOPatch.load(
            '{}/eopatch_{}'.format(path, patch_id),
            lazy_loading=True
        )

        _, height, width, _ = eopatch.data['BANDS'].shape
        mask = eopatch[mask_feature[0]][mask_feature[1]].squeeze()
        no_samples = min(height * width, no_samples)

        # Finds all the pixels which are not masked
        subsample_id = []
        for h in range(height):
            for w in range(width):
                # Check if pixel has any NaNs.
                has_nan = np.isnan(eopatch.data['BANDS'][:, h, w]).any()

                # Skip pixels with NaNs and masked pixels.
                if not has_nan and (mask is None or mask[h][w] == 1):
                    subsample_id.append((h, w))

        # First sampling
        subsample_id = random.sample(
            subsample_id,
            min(no_samples, len(subsample_id))
        )
        # print(f'Actual patch sample size: {len(subsample_id)}')

        for h, w in subsample_id:
            class_value = eopatch[class_feature[0]][class_feature[1]][h][w][0]

            array_for_dict = [(class_name, class_value)] \
                             + [(f[1], float(eopatch[f[0]][f[1]][h][w])) for f in features]

            if debug:
                array_for_dict += [('patch_no', patch_id), ('x', w), ('y', h)]
            sample_dict.append(dict(array_for_dict))

            # Enrichment
            if class_value in weak_classes:  # TODO check duplicates
                neighbours = [-3, -2, -1, 0, 1, 2, 3]
                for x in neighbours:
                    for y in neighbours:
                        if x != 0 or y != 0:
                            h0 = h + x
                            w0 = w + y
                            max_h, max_w = height, width
                            if h0 >= max_h or w0 >= max_w \
                                    or h0 <= 0 or w0 <= 0:
                                continue

                            val = eopatch[class_feature[0]][class_feature[1]][h0][w0][0]
                            if val in weak_classes:
                                array_for_dict = [(class_name, val)] \
                                                 + [(f[1], float(eopatch[f[0]][f[1]][h0][w0])) for f in features]
                                if debug:
                                    array_for_dict += [
                                        ('patch_no', patch_id),
                                        ('x', w0),
                                        ('y', h0)
                                    ]
                                sample_dict.append(dict(array_for_dict))

    df = pd.DataFrame(sample_dict, columns=columns)
    df.dropna(axis=0, inplace=True)

    class_dictionary = collections.Counter(df[class_feature[1]])
    class_count = class_dictionary.most_common()
    least_common = class_count[-1][1]
    # print(f'Least common: {least_common}')

    # Balancing
    replace = False
    if samples_per_class is not None:
        least_common = samples_per_class
        replace = True
    df_downsampled = pd.DataFrame(columns=columns)
    names = [name[0] for name in class_count]
    dfs = [df[df[class_name] == x] for x in names]
    for d in dfs:
        nd = resample(
            d,
            replace=replace,
            n_samples=least_common,
            random_state=seed
        )
        # print(f'Actual sample size per class: {len(nd.index)}')
        df_downsampled = df_downsampled.append(nd)

    if class_frequency:
        return df_downsampled, class_dictionary

    return df_downsampled


def get_patches(path, n=0):
    """Get selected number of patch IDs from given directory path.
    If number is not provided, i.e. is zero, all patch IDs are returned.

    :param path: Directory path where patches are
    :type path: Path
    :param n: Number of patch IDs to retrieve, defaults to 0
    :type n: int, optional
    :return: List of patch IDs
    :rtype: list[int]
    """
    patches = [patch.name for patch in path.glob('eopatch_*')]
    ids = []

    for patch in patches:
        match = re.match(r'^eopatch_(\d+)$', patch)
        if match:
            ids.append(int(match.group(1)))

    ids.sort()
    return random.sample(ids, n) if n else ids


if __name__ == '__main__':
    input_dir = 'E:/Data/PerceptiveSentinel/SVN/2017/processed/patches/'
    start_time = time.time()

    samples, class_dict = sample_patches(
        path=input_dir,
        patches=list(range(1085)),
        no_samples=20000,
        class_feature=(
            FeatureType.MASK_TIMELESS,
            'LPIS_2017_G2'
        ),
        mask_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
        features=[],
        weak_classes=[11, 5, 10, 6, 8, 9],
        debug=True,
        seed=None,
        class_frequency=True
    )
    sample_time = time.time() - start_time
    samples_path = 'D:/Samples/'
    filename = ''
    i = 0
    while True:
        filename = 'g2_samples' + str(i)
        if ospath.exists(samples_path + filename + '.csv'):
            i += 1
        else:
            break

    result = 'Class sample size: {0}. Sampling time {1}'.format(
        int(samples['LPIS_2017_G2'].size / pd.unique(samples['LPIS_2017_G2']).size), sample_time)
    print(result)
    print(class_dict)
    file = open('timing.txt', 'a')
    info = ' no_patches ' + str(1085) + ' samples_per_patch: ' + str(10000)
    dictionary = str(class_dict)
    file.write(
        '\n\n' + str(dt.datetime.now()) + ' SAMPLING ' + filename + ' ' + result + info + '\n' + dictionary + '\n')
    file.close()

    samples.to_csv(samples_path + filename + '.csv')
