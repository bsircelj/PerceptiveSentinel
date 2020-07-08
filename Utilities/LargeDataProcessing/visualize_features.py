import random
from eolearn.core import EOPatch, FeatureType

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Sampling import sample_patches
from PIL import Image
from sentinelhub import geo_utils
from scipy.ndimage import gaussian_filter
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import os


def display_rgb(patch_no, patches_path):
    x_size, y_size = patch_no.shape
    line = None
    image = None
    for xp in range(x_size):
        for yp in range(y_size):
            eopatch = EOPatch.load(f'{patches_path}/eopatch_{patch_no[xp, yp]}')
            image_add = np.clip(eopatch.data['BANDS'][15][..., [3, 2, 1]] * 3.5, 0, 1)
            if line is None:
                line = image_add
            else:
                line = np.concatenate((line, image_add), axis=1)
        if image is None:
            image = line
        else:
            image = np.concatenate((image, line), 0)
        line = None

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    patch_no = np.array([[398, 421, 433, 466],
                         [397, 420, 442, 465],
                         [396, 419, 441, 464],
                         [395, 418, 440, 463]])
    # patches_path = 'E:/Data/PerceptiveSentinel/SVN/2017/processed/patches'
    patches_path = 'D:/Users/Beno/Ljubljana2'
    feature_names = ['DEM', 'ARVI_max_mean_surf', 'ARVI_sd_val', 'NDVI_max_mean_len', 'NDVI_mean_val', 'NDVI_pos_len',
                     'SAVI_max_mean_feature', 'BLUE_max_mean_feature', 'GREEN_mean_val']
    class_feature = 'LPIS_2017_G2'

    img_path = 'D:/Users/Beno/Images/Ljubljana'
    x_size, y_size = patch_no.shape
    i = 0
    for xp in range(x_size):
        for yp in range(y_size):
            eopatch = EOPatch.load(path=f'{patches_path}/eopatch_{i}')
            img_path0 = f'{img_path}/patch_{i}'
            if not os.path.isdir(img_path0):
                os.makedirs(img_path0)
            for fet in feature_names:
                fig = plt.figure(fet)
                plt.imshow(eopatch.data_timeless[fet].squeeze())
                plt.savefig(f'{img_path0}/{fet}.png', dpi=300,
                            bbox_inches='tight')
                plt.close(fig)
            fig = plt.figure(class_feature)
            plt.imshow(eopatch.mask_timeless[class_feature].squeeze())
            plt.savefig(f'{img_path0}/{class_feature}.png', dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
            i+=1
