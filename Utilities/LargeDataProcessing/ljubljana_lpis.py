from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import plot_confusion_matrix
from joblib import dump, load
from eolearn.core import EOPatch, FeatureType, OverwritePermission
from eolearn.io import ExportToTiff
from all_stream_features import AddBaseFeatures
from temporal_features import AddStreamTemporalFeaturesTask
import sys
import numpy as np
import os
from eolearn.features import LinearInterpolation, SimpleFilterTask, LinearResampling, LegacyInterpolation, \
    InterpolationTask
import matplotlib
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from lj_process import apply_2d

class_color = {0: ('Not Farmland', 'xkcd:black'),
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

labels = [class_color[x][0] for x in class_color.keys()]

path = 'D:\\Users\\Beno\\'

feature_names = ['DEM', 'ARVI_max_mean_surf', 'ARVI_sd_val', 'NDVI_max_mean_len', 'NDVI_mean_val', 'NDVI_pos_len',
                 'SAVI_max_mean_feature', 'BLUE_max_mean_feature', 'GREEN_mean_val']
class_feature = 'LPIS_2017_G2'
no_classes = len(class_color)
names = []
boundaries = np.zeros(no_classes)

for i in range(no_classes):
    names.append(class_color[i][1])
    boundaries[i] = i - 0.5
cmap = matplotlib.colors.ListedColormap(names)
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

handles = []
for i in range(len(class_color)):
    patch = mpatches.Patch(color=class_color[i][1], label=class_color[i][0])
    handles.append(patch)


def add_legend(plot):
    legend = plot.legend(handles=handles, bbox_to_anchor=(1.5, 1.0), frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('gray')
    frame.set_edgecolor('black')


def create_geotiff(patch_no, patches_path, save_path, fet_name):
    x_size, y_size = patch_no.shape
    i = 0
    eopatch = EOPatch()
    line = None
    image = None
    for xp in range(x_size):
        for yp in range(y_size):
            print(f'___ Patch {xp * 3 + yp} ____')
            eopatch0 = EOPatch.load(f'{patches_path}/eopatch_{patch_no[xp, yp]}')
            image_add = eopatch0.data_timeless[fet_name].squeeze()
            if line is None:
                line = image_add
            else:
                line = np.concatenate((line, image_add), axis=1)
        if image is None:
            image = line
        else:
            image = np.concatenate((image, line), 0)
        line = None
    eopatch.add_feature(FeatureType.DATA)

    x_fst, y_fst, x_snd, y_snd = BBox._to_tuple(bbox)
    self.min_x = min(x_fst, x_snd)
    self.max_x = max(x_fst, x_snd)
    self.min_y = min(y_fst, y_snd)
    self.max_y = max(y_fst, y_snd)


def read_patch(patches_path, patch_id, feature_names, class_feature):
    eopatch = EOPatch.load(f'{patches_path}/eopatch_{patch_id}')
    t, width, height, _ = eopatch.data['BANDS'].shape

    features = [(FeatureType.DATA_TIMELESS, x) for x in feature_names]

    sample_dict = []
    coords = []
    for w in range(width):
        for h in range(height):
            array_for_dict = [(f[1], float(eopatch[f[0]][f[1]][w][h])) for f in features]
            sample_dict.append(dict(array_for_dict))
            coords.append([str(w) + ' ' + str(h)])
    ds = pd.DataFrame(sample_dict, columns=feature_names)
    # print('patch features')
    # print(ds)
    x_patch = ds[feature_names].to_numpy()
    lpis = eopatch.mask_timeless[class_feature].squeeze()
    lpis = apply_2d(lpis, lambda x: 0 if np.isnan(x) else int(x + 1))
    y_test = np.reshape(lpis, -1)

    return x_patch, y_test, (width, height)


def predict_patches(model, patch_no, load_path, save_path):
    x_size, y_size = patch_no.shape
    line = None
    image = None
    line_test = None
    image_test = None
    i = 0
    for xp in range(x_size):
        for yp in range(y_size):
            print(f'___ Patch {patch_no[xp, yp]} ____')
            # eopatch = EOPatch.load(path=f'{load_path}/eopatch_{patch_no[xp, yp]}')
            x_patch, y_test, shape = read_patch(load_path, patch_no[xp, yp], feature_names, class_feature)

            image_add = model.predict(x_patch)
            image_add = np.reshape(image_add, shape)
            image_add_test = np.reshape(y_test, shape)
            if line is None:
                line = image_add
                line_test = image_add_test
            else:
                line = np.concatenate((line, image_add), axis=1)
                line_test = np.concatenate((line_test, image_add_test), axis=1)
            if image is None:
                image = line
                image_test = line_test
            else:
                image = np.concatenate((image, line), 0)
            image_test = np.concatenate((image_test, line_test), 0)
            # np.save(f'{path}Images\\Classification\\Predicted_{patch_no[xp, yp]}', image_add)
            # np.save(f'{path}Images\\Classification\\Test_{patch_no[xp, yp]}', image_add_test)
            # plt.figure(dpi=150, figsize=(12, 6))
            # plt.subplot(121)
            # plt.imshow(image, cmap=cmap, norm=norm)
            # plt.subplot(122)
            # plt.imshow(image_add_test, cmap=cmap, norm=norm)
            # add_legend(plt)
            # plt.tight_layout()
            # plt.show()
            # plt.close()
        line = None
        line_test = None

    np.save(f'{path}Images\\Classification\\{i}', image)
    plt.figure(dpi=150, figsize=(8, 20))
    plt.subplot(121)
    plt.imshow(image, cmap=cmap, norm=norm)
    plt.subplot(122)
    plt.imshow(image_test, cmap=cmap, norm=norm)
    add_legend(plt)
    plt.show()
    eopatch = EOPatch()
    eopatch.add_feature(FeatureType.DATA_TIMELESS, 'LPIS', image[..., np.newaxis])
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    tiff = ExportToTiff(feature=(FeatureType.DATA_TIMELESS, 'LPIS'),
                        folder=f'{path}Images/Ljubljana/Ljubljana.tif')
    tiff.execute(eopatch)
    eopatch.save(path=f'{save_path}/eopatch_all',
                 overwrite_permission=OverwritePermission.OVERWRITE_PATCH)


def visualize(eopatch):
    for i in range(len(eopatch.timestamp)):
        plt.figure(i, figsize=(18, 6))
        img = np.clip(eopatch.data['BANDS'][i][..., [3, 2, 1]] * 3.5, 0, 1)
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(eopatch.mask['CLM2'][i].squeeze(), cmap='gray')
        plt.subplot(133)
        plt.imshow(eopatch.mask['IS_VALID'][i].squeeze(), cmap='gray')
        plt.show()


if __name__ == '__main__':
    patch_no = np.array([[398, 421, 433, 466],
                         [397, 420, 442, 465],
                         [396, 419, 441, 464],
                         [395, 418, 440, 463]])
    patches_path = 'E:/Data/PerceptiveSentinel/SVN/2017/processed/patches'
    save_path = 'D:/Users/Beno/Ljubljana'
    save_path2 = 'D:/Users/Beno/Ljubljana2'

    # interpolation(save_path=save_path,
    #               patches_path=patches_path,
    #               patch_no=patch_no)
    #
    # check(398)
    #
    name = 'dt'
    # create_model(name)
    model = load(f'{path}models/{name}.joblib')

    # calculate_features(patch_no=patch_no, patches_path=save_path, save_path=save_path2)
    predict_patches(model, patch_no, save_path2, 'D:/Users/Beno/LJ_Predicted')
