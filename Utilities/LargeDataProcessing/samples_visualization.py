import matplotlib.pyplot as plt
import numpy as np
from data_sample import BalancedClassSampler
from eolearn.core import EOTask, EOPatch
import random


def color_patch(image, colors=None):
    # Just for visualization of segments
    w, h = image.shape
    if colors is None:
        labels = np.max(image)
        labels = 0 if np.isnan(labels) else int(labels)
        colors = np.array([[0, 0, 0]])
        for _ in range(labels + 40):
            n_col = np.array([[random.randint(15, 255), random.randint(15, 255), random.randint(15, 255)]])
            colors = np.concatenate((colors, n_col), axis=0)

    new_image = np.zeros((w, h, 3))
    for x in range(w):
        for y in range(h):
            a = image[x][y]
            a = 0 if np.isnan(a) else a + 1
            c = colors[int(a)]
            new_image[x][y] = c
            # new_image[x][y] = colors[image[x][y]]

    return new_image / 255


new_color = {0: ('Not Farmland', 'xkcd:black'),
             1: ('Grass', 'xkcd:green'),
             2: ('Maize', 'xkcd:butter'),
             3: ('Orchards', 'xkcd:red'),
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

crop_g2 = dict()
for k in new_color:
    crop_g2[k] = new_color[k][0]

normal_10k = {8.0: 715458, 7.0: 267291, 21.0: 186829, 6.0: 64277, 20.0: 47033, 10.0: 20509, 9.0: 18243,
              4.0: 16672, 14.0: 14660, 16.0: 14284, 22.0: 13211, 17.0: 10402, 5.0: 10261, 0.0: 9853, 13.0: 6822,
              19.0: 5540,
              11.0: 2294, 3.0: 2038, 2.0: 1724, 18.0: 1159, 15.0: 767, 1.0: 453, 12.0: 77}

enriched_15k = {8.0: 1074825, 7.0: 400348, 21.0: 280895, 6.0: 96509, 20.0: 69075, 11.0: 68361,
                3.0: 56038,
                2.0: 49532, 18.0: 36440, 10.0: 31218, 9.0: 27450, 4.0: 24684, 15.0: 23825, 14.0: 22201, 16.0: 21262,
                22.0: 19778,
                17.0: 15882, 5.0: 15687, 0.0: 14783, 1.0: 13093, 13.0: 10123, 19.0: 8476, 12.0: 2590}

crop_names = {0: 'Beans', 1: 'Beets', 2: 'Buckwheat', 3: 'Fallow land', 4: 'Grass', 5: 'Hop',
              6: 'Leafy Legumes', 7: 'Maize', 8: 'Meadows', 9: 'Orchards', 10: 'Other',
              11: 'Peas',
              12: 'Poppy', 13: 'Potatoes', 14: 'Pumpkins', 15: 'Soft fruits', 16: 'Soybean', 17: 'Summer cereals',
              18: 'Sun flower', 19: 'Vegetables', 20: 'Vineyards', 21: 'Winter cereals', 22: 'Winter rape'}

g2 = {4: 1830770, 12: 1562291, 1: 1125416, 2: 256666, 13: 205228, 3: 18578, 7: 17353, 14: 13713, 9: 11847, 8: 11844,
      6: 10909, 11: 8051, 5: 2816, 10: 1092}
g3 = {4: 3661382, 1: 2253431, 8: 875024, 9: 673194, 2: 515058, 11: 488227, 6: 461144, 13: 410877, 5: 164392, 12: 96859,
      10: 65348, 3: 37200, 7: 34355, 14: 27812}


def change_name(dic, dic2):
    named = dict()
    highest1 = dic[8.0]
    highest2 = dic2[8.0]
    print('{0:15}: {1:7}  {2:7}  {3:7}'.format('name', 'normal', 'enriched', 'difference'))
    for no in crop_names.keys():
        value = dic[no]
        named[new_color[no][0]] = value
        h1 = value / highest1
        h2 = dic2[no] / highest2
        diff = h2 - h1
        print('{0:15}: {1:6.5f}  {2:6.5f}  {3:6.5f}'.format(crop_names[no], h1, h2, diff))
    return named


def example_test():
    # test_patch = 'D:/Users/Beno/PycharmProjects/eo-learn/example_data'
    save_file = 'D:/Samples/example.csv'

    # sampling = BalancedClassSampler(class_feature='LULC',
    #                                 load_path=test_patch,
    #                                 patches=['TestEOPatch'],
    #                                 samples_amount=1,
    #                                 # seed=1234,
    #                                 valid_mask='RANDOM_UINT8',
    #                                 weak_classes=None)

    test_patch = 'D:/Users/Beno/PycharmProjects/eo-learn/features/eolearn/tests/TestInputs'
    sampling = BalancedClassSampler(class_feature='LULC',
                                    load_path=test_patch,
                                    patches=['TestPatch'],
                                    samples_amount=0.1,
                                    valid_mask='mask',
                                    ignore_labels=None,
                                    features=None,
                                    weak_classes=None,
                                    search_radius=3,
                                    samples_per_class=None,
                                    seed=None,
                                    class_frequency=False)

    samples, result = sampling()
    # samples.to_csv(save_file)
    plt.figure(figsize=(18, 6))
    plt.suptitle(len(samples.index))
    plt.subplot(131)
    eopatch = EOPatch.load(f'{test_patch}/TestPatch')
    # img = np.clip(eopatch.data['BANDS-S2-L1C'][4][..., [3, 2, 1]] * 3.5, 0, 1)
    # plt.imshow(img)
    # for index, row in samples.iterrows():
    #     x = row['x']
    #     y = row['y']
    #     plt.scatter(x, y, color='r')
    plt.imshow(eopatch.mask_timeless['RANDOM_UINT8'])

    plt.subplot(132)
    lulc = eopatch.mask_timeless['LULC'].squeeze()
    lulc = color_patch(lulc)
    plt.imshow(lulc)
    for index, row in samples.iterrows():
        x = row['x']
        y = row['y']
        plt.scatter(x, y, color='r')
    return result


def draw_histogram(distribution):
    named = dict()
    for no in distribution.keys():
        value = distribution[no]
        named[new_color[no][0]] = value

    distribution = named
    plt.subplot(133)
    plt.bar(range(len(distribution)), list(distribution.values()), align='center')
    plt.xticks(range(len(distribution)), list(distribution.keys()), rotation='vertical')
    plt.show()


if __name__ == '__main__':
    while True:
        dist = example_test()
        draw_histogram(dist)
