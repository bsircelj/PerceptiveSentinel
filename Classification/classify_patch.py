import matplotlib.pyplot as plt
from classification_comparison import create_dict, cluster_df
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Utilities.LargeDataProcessing.Sampling import sample_patches
from eolearn.core import FeatureType, EOPatch, OverwritePermission, EOTask
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.cluster.hierarchy as sch
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import matplotlib
import matplotlib.colors as colors
import numpy.ma as ma
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix
import matplotlib.patches as mpatches
from joblib import dump, load

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = '/home/beno/Documents/IJS/Perceptive-Sentinel/'

crop_names = {0: 'Not Farmland', 1: 'Beans', 2: 'Beets', 3: 'Buckwheat', 4: 'Fallow land', 5: 'Grass', 6: 'Hop',
              7: 'Legumes or grass', 8: 'Maize', 9: 'Meadows', 10: 'Orchards', 11: 'Other', 12: 'Peas', 13: 'Poppy',
              14: 'Potatoes', 15: 'Pumpkins', 16: 'Soft fruits', 17: 'Soybean', 18: 'Summer cereals', 19: 'Sun flower',
              20: 'Vegetables', 21: 'Vineyards', 22: 'Winter cereals', 23: 'Winter rape'}

class_names = [crop_names[x] for x in range(24)]

name_and_color = {0: ('Not Farmland', 'xkcd:black'),
                  1: ('Beans', 'xkcd:blue'),
                  2: ('Beets', 'xkcd:magenta'),
                  3: ('Buckwheat', 'xkcd:burgundy'),
                  4: ('Fallow land', 'xkcd:grey'),
                  5: ('Grass', 'xkcd:brown'),
                  6: ('Hop', 'xkcd:green'),
                  7: ('Legumes or grass', 'xkcd:yellow green'),
                  8: ('Maize', 'xkcd:butter'),
                  9: ('Meadows', 'xkcd:red'),
                  10: ('Orchards', 'xkcd:royal purple'),
                  11: ('Other', 'xkcd:white'),
                  12: ('Peas', 'xkcd:spring green'),
                  13: ('Poppy', 'xkcd:mauve'),
                  14: ('Potatoes', 'xkcd:poo'),
                  15: ('Pumpkins', 'xkcd:pumpkin'),
                  16: ('Soft fruits', 'xkcd:grapefruit'),
                  17: ('Soybean', 'xkcd:baby green'),
                  18: ('Summer cereals', 'xkcd:cool blue'),
                  19: ('Sun flower', 'xkcd:piss yellow'),
                  20: ('Vegetables', 'xkcd:bright pink'),
                  21: ('Vineyards', 'xkcd:grape'),
                  22: ('Winter cereals', 'xkcd:ice blue'),
                  23: ('Winter rape', 'xkcd:neon blue')}
joined_classes = [7, 5, 9]

names = []
boundaries = np.zeros(24)
for i in range(24):
    names.append(name_and_color[i][1])
    boundaries[i] = i - 0.5
cmap = matplotlib.colors.ListedColormap(names)
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

handles = []
for i in range(len(name_and_color)):
    patch = mpatches.Patch(color=name_and_color[i][1], label=name_and_color[i][0])
    handles.append(patch)

features_old = [(FeatureType.DATA_TIMELESS, 'ARVI_max_mean_len'),
                (FeatureType.DATA_TIMELESS, 'EVI_min_val'),
                (FeatureType.DATA_TIMELESS, 'NDVI_min_val'),
                (FeatureType.DATA_TIMELESS, 'NDVI_sd_val'),
                (FeatureType.DATA_TIMELESS, 'SAVI_min_val'),
                (FeatureType.DATA_TIMELESS, 'SIPI_mean_val')
                ]

features400 = [(FeatureType.DATA_TIMELESS, 'DEM'),
               (FeatureType.DATA_TIMELESS, 'ARVI_max_mean_len'),
               (FeatureType.DATA_TIMELESS, 'ARVI_max_mean_surf'),
               (FeatureType.DATA_TIMELESS, 'BLUE_mean_val'),
               (FeatureType.DATA_TIMELESS, 'GREEN_pos_surf')
               ]

results_path = '/home/beno/Documents/test/Results/'


def save_figure(plt, file_name):
    # plt.savefig(f'/home/beno/Documents/IJS/Perceptive-Sentinel/Images/Features/{file_name}', dpi=300,
    #             bbox_inches='tight')
    plt.savefig(f'D:/users/Beno/{file_name}', dpi=300,
                bbox_inches='tight')


def get_data(samples_path):
    dataset = pd.read_csv(samples_path)
    # dataset.drop(columns=['INCLINATION'])
    # dataset.drop(columns=['NDVI_min_val', 'SAVI_min_val', 'INCLINATION'])
    y = dataset['LPIS_2017'].to_numpy()
    # !!!! -1 is marking no LPIS data so everything is shifted by one cause some classifiers don't want negative numbers
    y = [a + 1 for a in y]

    feature_names = [t[1] for t in features400]
    # print('samples')
    # print(dataset[feature_names])
    x = dataset[feature_names].to_numpy()

    # dataset = sample_patches(path=path,
    #                          no_patches=6,
    #                          no_samples=10000,
    #                          class_feature=(FeatureType.MASK_TIMELESS, 'LPIS_2017'),
    #                          mask_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
    #                          features=features,
    #                          samples_per_class=1000,
    #                          debug=False,
    #                          seed=10222)
    return x, y


def fit_predict(x, y, model, labels, name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    no_classes = range(len(labels))
    fig, ax = plt.subplots()
    ax.set_ylim(bottom=0.14, top=0)
    ax.set_title(name)
    plot_confusion_matrix(model, x_test, y_test, labels=no_classes,
                          display_labels=labels,
                          cmap='viridis',
                          include_values=False,
                          xticks_rotation='vertical',
                          normalize='pred',
                          ax=ax)
    return y_pred, y_test


def form_clusters(y_test, y_pred, all_y, k=0.6):
    confusion = confusion_matrix(y_test, y_pred, normalize='pred')
    ds = pd.DataFrame(confusion, columns=class_names)
    dsc, ind = cluster_df(ds, k)
    ind = [x - 1 for x in ind]
    _, class_names_new = create_dict(ind, class_names)
    clustered_y = [ind[int(i)] for i in all_y]
    return clustered_y, class_names_new, ind


def create_model(model=None, name='Random forest', k=None):
    if model is None:
        model = tree.DecisionTreeClassifier()
    global class_names
    # x_samples, y_samples = get_data('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/genetic_samples1111.csv')
    x_samples, y_samples = get_data('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/genetic_samples1111.csv')

    renaming = None
    clustered_y = y_samples
    if k is not None:
        y_pred_samples, y_test_samples = fit_predict(x_samples, y_samples, model, class_names, name + str(1))
        clustered_y, class_names, renaming = form_clusters(y_pred_samples, y_test_samples, y_samples, k=k)
        fit_predict(x_samples, clustered_y, model, class_names, name + ' clustered')

    model.fit(x_samples, clustered_y)

    return model, renaming


def save_model(model, filename):
    dump(model, path + 'models/' + filename + '.joblib')


def load_model(filename):
    return load(path + 'models/' + filename + '.joblib')


def make_confusion(model, name, x_patch, y_test, stats):
    fig, ax = plt.subplots(dpi=150, figsize=(20, 20))
    plt.subplots_adjust(bottom=0.25)
    ax.set_ylim(bottom=0.14, top=0)
    plot_confusion_matrix(model, x_patch, y_test, labels=range(len(class_names)),
                          display_labels=class_names,
                          cmap='viridis',
                          include_values=False,
                          xticks_rotation='vertical',
                          normalize='pred',
                          ax=ax)
    ax.set_title(stats)
    save_figure(plt, name + '.png')


def apply_2d(lpis, func):
    width, height = lpis.shape
    for x in range(width):
        for y in range(height):
            lpis[x][y] = func(lpis[x][y])

    return lpis


def add_legend(plot):
    legend = plot.legend(handles=handles, bbox_to_anchor=(1.5, 1.0), frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('gray')
    frame.set_edgecolor('black')


def read_patch(patches_path, patch_id, features):
    eopatch = EOPatch.load('{}/eopatch_{}'.format(patches_path, int(patch_id)))
    t, width, height, _ = eopatch.data['BANDS'].shape

    feature_names = [t[1] for t in features]
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
    lpis = eopatch.mask_timeless['LPIS_2017'].squeeze()
    lpis = apply_2d(lpis, lambda x: 0 if np.isnan(x) else int(x + 1))
    y_test = np.reshape(lpis, -1)

    return x_patch, y_test, (width, height)

    # # Masking
    # mask = ma.masked_equal(lpis, 0)
    # mask = ma.getmask(mask)
    # mask = np.invert(mask)
    # mask = mask.astype(int)
    # print(mask)
    # img = pred * mask


class ClassifyPatchTask(EOTask):

    def __init__(self, model, feature_name):
        self.model = model
        self.feature_name = feature_name


if __name__ == '__main__':
    # model = tree.DecisionTreeClassifier()
    # name = 'dt'
    # model = RandomForestClassifier()
    name = 'Random forest nonmasked'

    # model, renaming = create_model(model)
    # save_model(model, name)
    model = load_model(name)

    # patches_path = '/home/beno/Documents/test/Slovenia'
    patches_path = 'E:/Data/PerceptiveSentinel/Slovenia'
    patch_id = 2

    x_patch, y_test, patch_shape = read_patch(patches_path, patch_id, features400)

    y_pred_patch = model.predict(x_patch)
    y_pred_patch = [joined_classes[0] if x in joined_classes else x for x in y_pred_patch]

    img = np.reshape(y_pred_patch, patch_shape)
    lpis = np.reshape(y_test, patch_shape)

    accuracy = accuracy_score(y_test, y_pred_patch)
    f1 = f1_score(y_test, y_pred_patch, labels=range(24), average='macro')
    stats = '{0:_<40} CA: {1:5.4f} F1: {2:5.4f}'.format(name, accuracy, f1)
    make_confusion(model, name, x_patch, y_test, stats)

    eopatch = EOPatch.load('{}/eopatch_{}'.format(patches_path, int(patch_id)))
    edges = eopatch.mask_timeless['EDGES_INV'].squeeze()

    # img = img*edges
    plt.figure(dpi=150, figsize=(12, 12))
    plt.title('Predicted')
    add_legend(plt)
    plt.imshow(img, cmap=cmap, norm=norm)
    save_figure(plt, name + ' predicted')

    # lpis = lpis*edges
    # apply_2d(lpis, lambda x: joined_classes[0] if x in joined_classes else x)
    plt.figure(dpi=150, figsize=(12, 12))
    plt.title('LPIS')
    add_legend(plt)
    plt.imshow(lpis, cmap=cmap, norm=norm)
    save_figure(plt, str(patch_id) + ' LPIS')
    plt.show()
