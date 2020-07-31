import collections
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
from eolearn.core import EOExecutor, EOPatch, FeatureType, LinearWorkflow, \
    LoadTask, OverwritePermission, SaveTask, FeatureParser
from eolearn.geometry import PointSamplingTask
from sklearn.utils import resample
import os
import datetime as dt
import matplotlib.pyplot as plt


class BalancedClassSampler:
    """
    A class that samples points from multiple patches and return a balanced set depending on the class label.
    This is done by sampling on each patch the desired amount and then balancing the data based on the smallest class
    or amount. If the amount is provided and there are classes with less than that number of points, random
    point are duplicated to reach the necessary size.
    """

    def __init__(self, class_feature, load_path, patches=None, samples_amount=0.1, valid_mask=None,
                 ignore_labels=None, features=None, weak_classes=None, search_radius=3,
                 samples_per_class=None, seed=None, class_frequency=False):
        """
        :param class_feature: Feature that contains class labels
        :type class_feature: (FeatureType, string) or string
        :param load_path: Path to folder containing all patches
        :type load_path: string
        :param patches: List of patch names or None for all patches in the directory
        :type patches: list[str]
        :param samples_amount: Number of samples taken per patch. If the number is on the interval [0...1] then that
        percentage of all points is taken. If the value is 1 all eligible points are taken.
        :type samples_amount: float
        :param valid_mask: Feature that defines the area from where samples are
            taken, if None the whole image is used
        :type valid_mask: (FeatureType, string), string or None
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
        self.patches = patches if patches else [name for name, _ in os.walk(load_path)]
        self.samples_amount = samples_amount
        self.valid_mask = next(
            FeatureParser(valid_mask, default_feature_type=FeatureType.MASK_TIMELESS)()) if valid_mask else None
        self.ignore_labels = ignore_labels
        self.features = FeatureParser(features, default_feature_type=FeatureType.DATA_TIMELESS) if features else None
        self.columns = [self.class_feature[1]] + ['patch_name', 'x', 'y']
        if features:
            self.columns += [x[1] for x in self.features]

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
            mask = eopatch[self.valid_mask].squeeze() if self.valid_mask else None
            total_points = height * width
            no_samples = self.samples_amount if self.samples_amount > 1 else int(total_points * self.samples_amount)
            no_samples = min(total_points, no_samples)

            # Finds all the pixels which are not masked
            subsample_id = []
            for loc_h in range(height):
                for loc_w in range(width):
                    if mask is not None or mask[loc_h][loc_w]:
                        subsample_id.append((loc_h, loc_w))

            # First sampling
            subsample_id = random.sample(
                subsample_id,
                min(no_samples, len(subsample_id))
            )
            # print(f'Actual patch sample size: {len(subsample_id)}')

            for loc_h, loc_w in subsample_id:
                class_value = eopatch[self.class_feature][loc_h][loc_w][0]
                if self.ignore_labels and class_value in self.ignore_labels:
                    continue

                array_for_dict = [(self.class_feature[1], class_value)] + [('patch_name', patch_name), ('x', loc_w),
                                                                           ('y', loc_h)]
                if self.features:
                    array_for_dict += [(f[1], float(eopatch[f][loc_h][loc_w])) for f in self.features]

                sample_dict.append(dict(array_for_dict))

                if self.weak_classes and self.search_radius is not 0:
                    sample_dict = self.local_enrichment(class_value, loc_h, loc_w, height,
                                                        width, eopatch, patch_name, sample_dict)

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
        # df_downsampled.to_csv(self.save_file, index=False)
        return df_downsampled, dict(class_dictionary)

    def local_enrichment(self, class_value, loc_h, loc_w, height, width, eopatch, patch_name, sample_dict):

        # Enrichment
        if class_value in self.weak_classes:
            neighbours = list(range(-self.search_radius, self.search_radius + 1))
            for x in neighbours:
                for y in neighbours:
                    if x != 0 or y != 0:
                        search_h = loc_h + x
                        search_w = loc_w + y
                        max_h, max_w = height, width
                        if search_h >= max_h or search_w >= max_w \
                                or search_h <= 0 or search_w <= 0:
                            continue

                        val = eopatch[self.class_feature][search_h][search_w][0]
                        if val in self.weak_classes:
                            array_for_dict = [(self.class_feature[1], val)] \
                                             + [('patch_name', patch_name), ('x', search_w), ('y', search_h)]
                            if self.features:
                                array_for_dict += [(f[1], float(eopatch[f][search_h][search_w])) for f in self.features]
                            array_for_dict = dict(array_for_dict)
                            if array_for_dict not in sample_dict:
                                sample_dict.append(dict(array_for_dict))
        return sample_dict


if __name__ == '__main__':
    class_feature = 'LPIS_2017_G2'
    load_path = 'E:/Data/PerceptiveSentinel/SVN/2017/processed/patches'
    patches = [f'eopatch_{x}' for x in range(500, 506)]
    seed = 1234
    sampling = BalancedClassSampler(class_feature=class_feature,
                                    load_path=load_path,
                                    patches=patches,
                                    samples_amount=0.3,
                                    seed=1234,
                                    valid_mask='EDGES_INV',
                                    weak_classes=list(range(5, 13)))
    class_dict = sampling()
    print(class_dict)
