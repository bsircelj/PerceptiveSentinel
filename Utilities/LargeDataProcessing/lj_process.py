from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix
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

class_color = {0: ('Not Farmland', 'xkcd:black'),
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

labels = [class_color[x][0] for x in class_color.keys()]

path = 'D:\\Users\\Beno\\'

feature_names = ['DEM', 'ARVI_max_mean_surf', 'ARVI_sd_val', 'NDVI_max_mean_len', 'NDVI_mean_val', 'NDVI_pos_len',
                 'SAVI_max_mean_feature', 'BLUE_max_mean_feature', 'GREEN_mean_val']
class_feature = 'LPIS_2017_G2'


def create_model(name):
    samples_name = 'final_g2_0.csv'
    dataset = pd.read_csv(f'D:/Samples/{samples_name}')
    y = dataset['LPIS_2017_G2'].to_numpy()
    x = dataset[feature_names].to_numpy()
    model = tree.DecisionTreeClassifier()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model.fit(x_train, y_train)
    dump(model, f'{path}models/{name}.joblib')

    y_pred_patch = model.predict(x_test)
    f1 = f1_score(y_test, y_pred_patch, labels=range(24), average='macro')
    stats = '{0:_<20} F1: {1:5.4f}'.format(name, f1)

    no_classes = range(len(labels))
    fig, ax = plt.subplots()
    ax.set_ylim(bottom=0.14, top=0)
    ax.set_title(stats)
    plot_confusion_matrix(model, x_test, y_test, labels=no_classes,
                          display_labels=labels,
                          cmap='viridis',
                          include_values=False,
                          xticks_rotation='vertical',
                          normalize='pred',
                          ax=ax)
    plt.savefig(f'D:/users/Beno/Images/{name}_confusion', dpi=300,
                bbox_inches='tight')

    plt.show()


needed_base = ['ARVI', 'NDVI', 'SAVI', 'BLUE', 'GREEN']


def calculate_features(patch_no, patches_path, save_path):
    x_size, y_size = patch_no.shape
    i = 0
    for xp in range(x_size):
        for yp in range(y_size):
            print(f'___ Patch {xp * 3 + yp} ____')
            eopatch = EOPatch.load(f'{patches_path}/eopatch_{patch_no[xp, yp]}')
            t, h, w, b = eopatch.data['BANDS'].shape
            eopatch.add_feature(FeatureType.MASK, 'IS_VALID', np.full((t, h, w, 1), True))
            base = AddBaseFeatures()
            eopatch = base.execute(eopatch)

            for n in needed_base:
                print(f'calculating {n}')
                add_stream = AddStreamTemporalFeaturesTask(data_feature=n)
                eopatch = add_stream.execute(eopatch)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            eopatch.save(path=f'{save_path}/eopatch_{patch_no[xp, yp]}',
                         overwrite_permission=OverwritePermission.OVERWRITE_PATCH
                         )
            i += 1


def check(no):
    # patches_path = 'E:/Data/PerceptiveSentinel/SVN/2017/processed/patches/eopatch_420'
    patches_path = f'D:/Users/Beno/Ljubljana/eopatch_{no}'
    eopatch = EOPatch.load(patches_path)
    for i in range(len(eopatch.timestamp)):
        plt.figure(i, figsize=(18, 6))
        img = np.clip(eopatch.data['BANDS'][i][..., [3, 2, 1]] * 3.5, 0, 1)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(eopatch.mask['CLM2'][i].squeeze(), cmap='gray')
        plt.show()


def apply_2d(lpis, func):
    width, height = lpis.shape
    for x in range(width):
        for y in range(height):
            lpis[x][y] = func(lpis[x][y])

    return lpis


def make_nan_mask(eopatch, fet_name='CLM2'):
    t, w, h, _ = eopatch.data['BANDS'].shape
    nan_mask = np.ones((t, w, h))
    for t0 in range(t):
        nan_mask[t0] = apply_2d(eopatch.data['BANDS'][t0, ..., 0], lambda x: 0 if np.isnan(x) else 1)
    eopatch.add_feature(FeatureType.MASK, fet_name, nan_mask[..., np.newaxis])
    return eopatch


def interpolation(patch_no, patches_path, save_path):
    copied_features = [(FeatureType.DATA, 'BANDS'),
                       (FeatureType.DATA_TIMELESS, 'DEM'),
                       (FeatureType.MASK_TIMELESS, 'LPIS_2017_G2'),
                       (FeatureType.MASK_TIMELESS, 'LULC_2017'),
                       (FeatureType.MASK_TIMELESS, 'LULC_2017_E'),
                       (FeatureType.MASK_TIMELESS, 'LULC_2017_G'),
                       (FeatureType.MASK_TIMELESS, 'LULC_2017_G_E')]

    linear_interp = InterpolationTask(feature=(FeatureType.DATA, 'BANDS'),
                                      copy_features=copied_features,
                                      interpolation_object=interpolate.interp1d,
                                      mask_feature=(FeatureType.MASK, 'CLM2'),
                                      interpolate_pixel_wise=True,
                                      kind='nearest',
                                      bounds_error=False,
                                      fill_value='extrapolate'
                                      )

    for i in patch_no:
        print(f'Processing {i}')
        eopatch = EOPatch.load(path=f'{patches_path}/eopatch_{patch_no[i]}')
        eopatch = make_nan_mask(eopatch)
        eopatch = linear_interp.execute(eopatch)
        save_path_location = f'{save_path}/eopatch_{patch_no[i]}'
        if not os.path.isdir(save_path_location):
            os.makedirs(save_path_location)
        eopatch.save(path=save_path_location,
                     features=copied_features,
                     overwrite_permission=OverwritePermission.OVERWRITE_PATCH)


if __name__ == '__main__':
    patch_no = np.array([[398, 421, 443, 466],
                         [397, 420, 442, 465],
                         [396, 419, 441, 464],
                         [395, 418, 440, 463]])

    # patches_path = 'E:/Data/PerceptiveSentinel/SVN/2017/processed/patches'
    patches_path = 'E:/Data/PerceptiveSentinel/SVN_Interpolated2'
    save_path = 'D:/Users/Beno/Ljubljana'

    calculate_features(patch_no, patches_path, save_path)
    # create_model('dt')
