import random
from eolearn.core import EOPatch, FeatureType

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Sampling import sample_patches
from PIL import Image
from sentinelhub import geo_utils


def save_figure(plt, file_name):
    plt.savefig(f'/home/beno/Documents/IJS/Perceptive-Sentinel/Images/Features/{file_name}', dpi=300,
                bbox_inches='tight')


def color_patch(image, colors=None):
    # Just for visualization of segments
    w, h = image.shape
    print(image.shape)
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


def display():
    path = '/home/beno/Documents/test'
    # path = 'E:/Data/PerceptiveSentinel'
    patch_no = 2
    eopatch = EOPatch.load(path + '/Slovenia/eopatch_{}'.format(patch_no), lazy_loading=True)
    # d = FeatureType.MASK_TIMELESS
    # f = 'EDGES_PRETTY'
    # print(len(eopatch.timestamp))
    # f = 'BLUE_mean_val'
    # d = FeatureType.DATA_TIMELESS
    # img = eopatch[d][f].squeeze()
    # d = FeatureType.DATA
    # img = eopatch[d][f][16].squeeze()
    img = np.clip(eopatch.data['BANDS'][10][..., [3, 2, 1]] * 3.5, 0, 1)

    # plt.title(f)
    plt.imshow(img)
    # save_figure(plt, f + '.png')
    plt.show()
    # save_figure(plt, f + '.png')
    # print(eopatch)
    # t, w, h = eopatch.data['BANDS'].squeeze()
    # res = geo_utils.bbox_to_resolution(eopatch.bbox, 337, 333)
    # print(str(res[0] * 337 )+ ' ' + str(res[1] * 333))
    # features = [(FeatureType.DATA_TIMELESS, 'SAVI_mean_val'),
    #             (FeatureType.DATA_TIMELESS, 'EVI_min_val'),
    #             (FeatureType.DATA_TIMELESS, 'EVI_sd_val'),
    #             (FeatureType.DATA_TIMELESS, 'NDVI_min_val'),
    #             (FeatureType.DATA_TIMELESS, 'NDVI_sd_val'),
    #             (FeatureType.DATA_TIMELESS, 'SAVI_min_val'),
    #             (FeatureType.DATA_TIMELESS, 'SIPI_mean_val'),
    #             (FeatureType.DATA_TIMELESS, 'ARVI_max_mean_len'),
    #             (FeatureType.DATA_TIMELESS, 'ARVI_max_mean_surf'),
    #             (FeatureType.DATA_TIMELESS, 'NDVI_max_mean_feature')
    #             ]
    #
    # # features = [f[1] for f in features]
    #
    features = [(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
                (FeatureType.DATA_TIMELESS, 'EVI_min_val'),
                (FeatureType.DATA_TIMELESS, 'EVI_sd_val'),
                ]

    for d, f in features:
        img = eopatch[d][f].squeeze()
        # plt.title(f)
        plt.imshow(img)
        save_figure(plt, f + '.png')

    # ax1.imshow(seg, cmap='gray')
    # ax1.imshow(mask, cmap=cmap, alpha=0.8)

    # plt.imshow(seg)
    # plt.show()
    # print(seg.shape)
    # plt.imshow(Image.blend(Image.fromarray(color_patch(seg)), Image.fromarray(seg), alpha=0.5))
    # elevation = eopatch.mask_timeless['EDGES_INV'].squeeze()
    # incline = eopatch.data_timeless['INCLINATION'].squeeze()
    # class_feature = np.squeeze(eopatch.mask_timeless['LPIS_2017'])
    #
    # cmap = matplotlib.colors.ListedColormap(np.random.rand(23, 3))
    # n_time = 5
    # image = np.clip(eopatch.data['BANDS'][n_time][..., [3, 2, 1]] * 3.5, 0, 1)
    # # mask = np.squeeze(eopatch.mask_timeless['LPIS_2017'])
    # name = 'LPIS_2017'
    # plt.title(name)
    # plt.imshow(image)
    # plt.imshow(class_feature, cmap=cmap)
    # save_figure(plt, name + '.png')

    # fig, ax1 = plt.subplots(figsize=(40, 40))
    # ax0.imshow(image)
    # print(seg)
    # ax1.imshow(elevation)
    # ax2.imshow(incline)
    # print(image)
    # seg = seg*255
    # image[:, :, 0] = image[:, :, 0] * seg
    # image[:, :, 1] = image[:, :, 1] * seg
    # image[:, :, 2] = image[:, :, 2] * seg
    # ax1.imshow(image)
    # ax1.imshow(seg, cmap='gray')
    # ax1.imshow(mask, cmap=cmap, alpha=0.8)
    # ax1.imshow(seg, cmap='gray', alpha=0.2)
    # ax1.imshow(seg)

    # path = '/home/beno/Documents/test/Slovenia'
    #
    # no_patches = patch_no + 1
    # no_samples = 10000
    # class_feature = (FeatureType.MASK_TIMELESS, 'LPIS_2017')
    # mask = (FeatureType.MASK_TIMELESS, 'EDGES_INV')
    # features = [(FeatureType.DATA_TIMELESS, 'NDVI_mean_val'), (FeatureType.DATA_TIMELESS, 'SAVI_max_val'),
    #             (FeatureType.DATA_TIMELESS, 'NDVI_pos_surf')]
    # samples_per_class = 10
    # debug = True
    #
    # samples = sample_patches(path, no_patches, no_samples, class_feature, mask, features, samples_per_class, debug)
    # print(samples)
    # for index, row in samples.iterrows():
    #    ax1.plot(row['x'], row['y'], 'ro', alpha=1)
    # save_figure(plt, name + '.png')
    # plt.show()


if __name__ == '__main__':
    display()
