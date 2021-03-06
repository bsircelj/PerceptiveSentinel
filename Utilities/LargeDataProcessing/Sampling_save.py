from eolearn.core import EOPatch, FeatureType
import numpy as np
from sklearn.utils import resample
import random
import pandas as pd
import collections
import time
import datetime as dt
import random
import os.path as pth


def sample_patches(path, no_patches, no_samples, class_feature, features, weak_classes=None, mask_feature=None,
                   samples_per_class=None,
                   debug=False, seed=None, class_frequency=False):
    """
    :param path: Path to folder containing all patches, folders need to be named eopatch_{number: 0 to no_patches-1}
    :param no_patches: Total number of patches
    :param no_samples: Number of samples taken per patch
    :param class_feature: Name of feature that contains class number.
        The numbers in the array can be float or nan for no class
    :type class_feature: (FeatureType, string)
    :param features: Features to include in returned dataset for each pixel sampled
    :type features: array of type [(FeatureType, string), ...]
    :param mask_feature: Feature that defines the area from where samples are taken, if None the whole image is used
    :type mask_feature: (FeatureType, String) or None
    :param samples_per_class: Number of samples per class returned after balancing. If the number is higher than minimal
        number of samples for the smallest class then those numbers are upsampled by repetition.
        If the argument is None then number is set to the size of the number of samples of the smallest class
    :type samples_per_class: int or None
    :param debug: If set to True patch id and coordinates are included in returned DataFrame
    :param seed: Seed for random generator
    :return: pandas DataFrame with columns [class feature, features, patch_id, x coord, y coord].
        id,x and y are used for testing
    :param class_frequency: If set to True, the function also return dictionary of each class frequency before balancing
    :type class_frequency: boolean
    :param weak_classes: Classes that when found also the neighbouring regions will be checked and added if they contain
        one of the weak classes. Used to enrich the samples
    :type weak_classes: int list
    """
    if seed is not None:
        random.seed(seed)
    columns = [class_feature[1]] + [x[1] for x in features]
    if debug:
        columns = columns + ['patch_no', 'x', 'y']
    class_name = class_feature[1]
    sample_dict = []

    for patch_id in range(no_patches):
        patch_id = 398
        path1 = '{}/eopatch_{}/mask_timeless/LPIS_2017.npy'.format(path, patch_id)
        path2 = '{}/eopatch_{}'.format('E:\\Data\\PerceptiveSentinel\\Slovenia_S1', patch_id)
        if not pth.exists(path1) or not pth.exists(path2):
            print('Patch {} missing.'.format(patch_id))
            continue
        print(patch_id)
        eopatch = EOPatch.load('{}/eopatch_{}'.format(path, patch_id), lazy_loading=True)
        _, height, width, _ = eopatch.data['BANDS'].shape
        # height, width = 500, 500  # Were supposed to be 505 and 500, but INCLINATION feature has wrong dimensions
        mask = eopatch[mask_feature[0]][mask_feature[1]].squeeze()
        mask = mask[0:height, 0:width]
        no_samples = min(height * width, no_samples)

        # Finds all the pixels which are not masked
        if mask_feature is None:
            mask = np.ones((height, width))
        stacked = np.stack(np.where(mask), axis=-1)
        for h, w in stacked:
            class_value = float(-1)
            if class_feature in eopatch.get_feature_list():
                val = float(eopatch[class_feature[0]][class_feature[1]][h][w])
                if not np.isnan(val):
                    class_value = val

            array_for_dict = [(class_name, class_value)] + [(f[1], float(eopatch[f[0]][f[1]][h][w])) for f in features]
            if debug:
                array_for_dict += [('patch_no', patch_id), ('x', w), ('y', h)]
            sample_dict.append(dict(array_for_dict))

            # Enrichment
            if weak_classes is not None:
                if class_value in weak_classes:  # TODO check duplicates
                    neighbours = [-3, -2, -1, 0, 1, 2, 3]
                    for x in neighbours:
                        for y in neighbours:
                            if x != 0 or y != 0:
                                h0 = h + x
                                w0 = w + y
                                max_h, max_w = 500, 500
                                if h0 >= max_h or w0 >= max_w or h0 <= 0 or w0 <= 0:
                                    continue
                                val = float(eopatch[class_feature[0]][class_feature[1]][h0][w0])
                                if val in weak_classes:
                                    array_for_dict = [(class_name, val)] + [(f[1], float(eopatch[f[0]][f[1]][h0][w0]))
                                                                            for f in features]
                                    if debug:
                                        array_for_dict += [('patch_no', patch_id), ('x', w0), ('y', h0)]
                                    sample_dict.append(dict(array_for_dict))
        break

    df = pd.DataFrame(sample_dict, columns=columns)
    df.dropna(axis=0, inplace=True)
    # Change to new class mapping
    df[class_feature[1]] = [mapping[x] for x in df[class_feature[1]]]

    class_dictionary = collections.Counter(df[class_feature[1]])
    class_count = class_dictionary.most_common()
    least_common = class_count[-1][1]

    # Balancing
    replace = False
    if samples_per_class is not None:
        least_common = samples_per_class
        replace = True
    df_downsampled = pd.DataFrame(columns=columns)
    names = [name[0] for name in class_count]
    dfs = [df[df[class_name] == x] for x in names]
    for d in dfs:
        nd = resample(d, replace=replace, n_samples=least_common, random_state=seed)
        df_downsampled = df_downsampled.append(nd)

    if class_frequency:
        return df_downsampled, class_dictionary
    return df_downsampled


new_classes = {0: ('Not Farmland', 'xkcd:black'),
               1: ('Grass', 'xkcd:brown'),
               2: ('Maize', 'xkcd:butter'),
               3: ('Orchards', 'xkcd:royal purple'),
               4: ('Other', 'xkcd:white'),
               5: ('Peas', 'xkcd:spring green'),
               6: ('Potatoes', 'xkcd:poo'),
               7: ('Pumpkins', 'xkcd:pumpkin'),
               8: ('Soybean', 'xkcd:baby green'),
               9: ('Summer cereals', 'xkcd:cool blue'),
               10: ('Sun flower', 'xkcd:piss yellow'),
               11: ('Vegetables', 'xkcd:bright pink'),
               12: ('Vineyards', 'xkcd:grape'),
               13: ('Winter cereals', 'xkcd:ice blue'),
               14: ('Winter rape', 'xkcd:neon blue')}

mapping = {0: 0,
           1: 4,  ####
           2: 4,  ###
           3: 4,  ###
           4: 4,  ###
           5: 1,
           6: 4,  ###
           7: 1,  ###
           8: 2,
           9: 1,  ###
           10: 3,
           11: 4,
           12: 5,
           13: 4,  ###
           14: 6,
           15: 7,
           16: 3,  ###
           17: 8,
           18: 9,
           19: 10,
           20: 11,
           21: 12,
           22: 13,
           23: 14}

old_classes = {0: ('Not Farmland', 'xkcd:black'),
               1: ('Beans', 'xkcd:blue'),  ####
               2: ('Beets', 'xkcd:magenta'),  ###
               3: ('Buckwheat', 'xkcd:burgundy'),  ###
               4: ('Fallow land', 'xkcd:grey'),  ###
               5: ('Grass', 'xkcd:brown'),
               6: ('Hop', 'xkcd:green'),  ###
               7: ('Legumes or grass', 'xkcd:yellow green'),  ###
               8: ('Maize', 'xkcd:butter'),
               9: ('Meadows', 'xkcd:red'),  ###
               10: ('Orchards', 'xkcd:royal purple'),
               11: ('Other', 'xkcd:white'),
               12: ('Peas', 'xkcd:spring green'),
               13: ('Poppy', 'xkcd:mauve'),  ###
               14: ('Potatoes', 'xkcd:poo'),
               15: ('Pumpkins', 'xkcd:pumpkin'),
               16: ('Soft fruits', 'xkcd:grapefruit'),  ###
               17: ('Soybean', 'xkcd:baby green'),
               18: ('Summer cereals', 'xkcd:cool blue'),
               19: ('Sun flower', 'xkcd:piss yellow'),
               20: ('Vegetables', 'xkcd:bright pink'),
               21: ('Vineyards', 'xkcd:grape'),
               22: ('Winter cereals', 'xkcd:ice blue'),
               23: ('Winter rape', 'xkcd:neon blue')}

# Example of usage
if __name__ == '__main__':
    patches_path = 'E:/Data/PerceptiveSentinel/Slovenia'
    # patches_path = '/home/beno/Documents/test/Slovenia'

    start_time = time.time()
    no_patches = 1060
    no_samples = 10000
    # eopatch = EOPatch.load('{}/eopatch_{}'.format(patches_path, 500), lazy_loading=True)
    # all_features = eopatch.get_feature_list()
    # all_data_timeless = []
    # for f in all_features[0:128]:
    #     if len(f) != 2:
    #         continue
    #     if f[0] == FeatureType.DATA_TIMELESS:
    #         all_data_timeless.append((f[0], f[1]))

    samples, class_dict = sample_patches(path=patches_path,
                                         no_patches=no_patches,
                                         no_samples=no_samples,
                                         class_feature=(FeatureType.MASK_TIMELESS, 'LPIS_2017'),
                                         mask_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
                                         features=[],
                                         debug=True,
                                         seed=None,
                                         class_frequency=True,
                                         weak_classes=[19, 12, 20])

    sample_time = time.time() - start_time
    filename = 'enriched_samples' + str(int(random.random() * 10000))
    # print(samples)
    result = 'Class sample size: {0}. Sampling time {1}'.format(
        int(samples['LPIS_2017'].size / pd.unique(samples['LPIS_2017']).size), sample_time)
    print(result)
    print(class_dict)
    file = open('timing.txt', 'a')
    info = ' no_patches ' + str(no_patches) + ' samples_per_patch: ' + str(no_samples)
    dictionary = str(class_dict)
    file.write(
        '\n\n' + str(dt.datetime.now()) + ' SAMPLING ' + filename + ' ' + result + info + '\n' + dictionary + '\n')
    file.close()

    samples.to_csv('D:/Samples/' + filename + '.csv')
