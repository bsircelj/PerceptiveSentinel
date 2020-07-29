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
    LoadTask, OverwritePermission, SaveTask, FeatureParser
from eolearn.geometry import PointSamplingTask
from sklearn.utils import resample
import os
import datetime as dt


class BalancedClassSampler:
    """
    A class that samples points from multiple patches and return a balanced set depending on the class label.
    This is done by sampling on each patch the desired amount and then balancing the data based on the smallest class
    or amount. If the amount is provided and there are classes with less than that number of points, random
    point are duplicated to reach the necessary size.

    It also supports additional sampling around specified weak classes.
    """

    def __init__(self, class_feature, load_path, save_file, patches=None, samples_amount=0.1, valid_mask=None,
                 ignore_labels=None, features=None, weak_classes=None, search_radius=3,
                 samples_per_class=None, seed=None, class_frequency=False):
        """
        :param class_feature: Feature that contains class labels
        :type class_feature: (FeatureType, string)
        :param load_path: Path to folder containing all patches
        :type load_path: string
        :param save_file: Location and name of file to where samples are saved
        :type save_file: string
        :param patches: List of patch names or None for all patches in the directory
        :type patches: list[str]
        :param samples_amount: Number of samples taken per patch. If the number is on the interval [0...1] then that
        percentage of all points is taken
        :type samples_amount: float
        :param valid_mask: Feature that defines the area from where samples are
            taken, if None the whole image is used
        :type valid_mask: (FeatureType, String) or None
        :param ignore_labels: A list of label values that should not be sampled.
        :type ignore_labels: list of integers
        :param features: temporal Features to include in dataset for each pixel sampled
        :type features: array of type [(FeatureType, string), ...]
        :param samples_per_class: Number of samples per class returned after
            balancing. If the number is higher than minimal number of samples for
            the smallest class then those numbers are upsampled by repetition.
            If the argument is None then number is set to the size of the number of
            samples of the smallest class
        :type samples_per_class: int or None
        :param seed: Seed for random generator
        :type class_frequency: boolean
        :param weak_classes: Classes that when found also the neighbouring regions
            will be checked and added if they contain one of the weak classes.
            Used to enrich the samples
        :param search_radius: How many points in each direction to check for additional weak classes
        :type search_radius: int
        :type weak_classes: int list
        :return: pandas DataFrame with columns
            [class feature, features, patch_id, x coord, y coord].
        """
        self.class_feature = next(FeatureParser(class_feature, default_feature_type=FeatureType.MASK_TIMELESS)())
        self.load_path = load_path
        self.save_file = save_file
        self.patches = patches if patches else [name for name, _ in os.walk(load_path)]
        self.samples_amount = samples_amount
        self.valid_mask = next(FeatureParser(valid_mask, default_feature_type=FeatureType.MASK_TIMELESS)())
        self.ignore_labels = ignore_labels
        self.features = FeatureParser(features, default_feature_type=FeatureType.DATA_TIMELESS)
        self.columns = [class_feature[1]] + [x[1] for x in self.features] + ['patch_name', 'x', 'y']

        self.samples_per_class = samples_per_class
        self.seed = random.seed(seed) if seed is not None else None
        self.class_frequency = class_frequency
        self.weak_classes = weak_classes
        self.search_radius = search_radius

    def __call__(self):

        sample_dict = []

        for patch_name in self.patches:
            eopatch = EOPatch.load(f'{self.load_path}/{patch_name}', lazy_loading=True)

            height, width, _ = eopatch[self.class_feature].shape
            mask = eopatch[self.valid_mask].squeeze()
            total_points = height * width
            no_samples = self.samples_amount if self.samples_amount >= 1 else total_points * self.samples_amount
            no_samples = min(total_points, no_samples)

            # Finds all the pixels which are not masked
            subsample_id = []
            for h in range(height):
                for w in range(width):
                    # Skip pixels with NaNs and masked pixels.
                    if mask is None or mask[h][w] == 1:
                        subsample_id.append((h, w))

            # First sampling
            subsample_id = random.sample(
                subsample_id,
                min(no_samples, len(subsample_id))
            )
            # print(f'Actual patch sample size: {len(subsample_id)}')

            for h, w in subsample_id:
                class_value = eopatch[self.class_feature][h][w][0]
                if class_value in self.ignore_labels:
                    continue

                array_for_dict = [(self.class_feature[1], class_value)] \
                                 + [(f[1], float(eopatch[f[0]][f[1]][h][w])) for f in self.features] \
                                 + [('patch_name', patch_name), ('x', w), ('y', h)]

                sample_dict.append(dict(array_for_dict))

                sample_dict = self.local_enrichment(class_value, height, width, eopatch, patch_name, sample_dict)

        df = pd.DataFrame(sample_dict, columns=self.columns)
        df.dropna(axis=0, inplace=True)

        class_dictionary = collections.Counter(df[self.class_feature[1]])
        class_count = class_dictionary.most_common()
        least_common = class_count[-1][1]
        # print(f'Least common: {least_common}')

        # Balancing
        replace = False
        if self.samples_per_class is not None:
            least_common = self.samples_per_class
            replace = True
        df_downsampled = pd.DataFrame(columns=self.columns)
        names = [name[0] for name in class_count]
        dfs = [df[df[self.class_feature[1]] == x] for x in names]
        for d in dfs:
            nd = resample(
                d,
                replace=replace,
                n_samples=least_common,
                random_state=self.seed
            )
            # print(f'Actual sample size per class: {len(nd.index)}')
            df_downsampled = df_downsampled.append(nd)
        # to file: class_dictionary

        return df_downsampled

    def local_enrichment(self, class_value, height, width, eopatch, patch_name, sample_dict):
        # Enrichment
        if class_value in self.weak_classes:
            neighbours = list(range(-self.search_radius, self.search_radius + 1))
            for x in neighbours:
                for y in neighbours:
                    if x != 0 or y != 0:
                        h0 = h + x
                        w0 = w + y
                        max_h, max_w = height, width
                        if h0 >= max_h or w0 >= max_w \
                                or h0 <= 0 or w0 <= 0:
                            continue

                        val = eopatch[self.class_feature][h0][w0][0]
                        if val in self.weak_classes:
                            array_for_dict = [(self.class_feature[1], val)] \
                                             + [(f[1], float(eopatch[f[0]][f[1]][h0][w0])) for f in
                                                self.features] \
                                             + [('patch_no', patch_name), ('x', w0), ('y', h0)]
                            array_for_dict = dict(array_for_dict)
                            if array_for_dict not in sample_dict:
                                sample_dict.append(dict(array_for_dict))
        return sample_dict


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
