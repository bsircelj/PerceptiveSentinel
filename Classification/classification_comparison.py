from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Utilities.LargeDataProcessing.Sampling import sample_patches
from eolearn.core import FeatureType
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

crop_names = {0: 'Beans', 1: 'Beets', 2: 'Buckwheat', 3: 'Fallow land', 4: 'Grass', 5: 'Hop',
              6: 'Legumes or grass', 7: 'Maize', 8: 'Meadows', 9: 'Orchards', 10: 'Other',
              11: 'Peas',
              12: 'Poppy', 13: 'Potatoes', 14: 'Pumpkins', 15: 'Soft fruits', 16: 'Soybean', 17: 'Summer cereals',
              18: 'Sun flower', 19: 'Vegetables', 20: 'Vineyards', 21: 'Winter cereals', 22: 'Winter rape'}
class_names = ['Not Farmland'] + [crop_names[x] for x in range(23)]

# features = [(FeatureType.DATA_TIMELESS, 'ARVI_max_mean_len'),
#             (FeatureType.DATA_TIMELESS, 'EVI_min_val'),
#             (FeatureType.DATA_TIMELESS, 'NDVI_min_val'),
#             (FeatureType.DATA_TIMELESS, 'NDVI_sd_val'),
#             (FeatureType.DATA_TIMELESS, 'SAVI_min_val'),
#             (FeatureType.DATA_TIMELESS, 'SIPI_mean_val')
#             ]

features1900 = [(FeatureType.DATA_TIMELESS, 'DEM'),
                (FeatureType.DATA_TIMELESS, 'ARVI_max_mean_len'),
                (FeatureType.DATA_TIMELESS, 'BLUE_max_mean_surf'),
                (FeatureType.DATA_TIMELESS, 'BLUE_mean_val'),
                (FeatureType.DATA_TIMELESS, 'BLUE_neg_surf')
                ]
features400 = [(FeatureType.DATA_TIMELESS, 'DEM'),
               (FeatureType.DATA_TIMELESS, 'ARVI_max_mean_len'),
               (FeatureType.DATA_TIMELESS, 'ARVI_max_mean_surf'),
               (FeatureType.DATA_TIMELESS, 'BLUE_mean_val'),
               (FeatureType.DATA_TIMELESS, 'GREEN_pos_surf')
               ]


def get_data(samples_path, feature_names):
    dataset = pd.read_csv(samples_path)
    # dataset.drop(columns=['INCLINATION'])
    # dataset.drop(columns=['NDVI_min_val', 'SAVI_min_val', 'INCLINATION'])
    y = dataset['LPIS_2017'].to_numpy()
    # !!!! -1 is marking no LPIS data so everything is shifted by one cause some classifiers don't want negative numbers
    y = [a + 1 for a in y]

    # feature_names = [t[1] for t in features400]
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


def save_figure(plt, file_name):
    plt.savefig(f'/home/beno/Documents/IJS/Perceptive-Sentinel/Images/Comparison/{file_name}', dpi=300,
                bbox_inches='tight')


def cluster_df(df, k=0.5):
    X = df.corr().values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, k * d.max(), 'distance')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)
    return df, ind


def create_dict(ind, group_names):
    new_dict = dict()
    for ni in range(max(ind) + 1):
        new_name = ''
        for i, names in enumerate(group_names):
            if ni == ind[i]:
                new_name += names + ', '
        new_dict[ni] = new_name[0:-2]
    no_classes_new = len(new_dict)
    class_names_new = [new_dict[x] for x in range(no_classes_new)]
    return new_dict, class_names_new


def form_clusters(y_test, y_pred, all_y, k=0.6):
    confusion = confusion_matrix(y_test, y_pred, normalize='pred')
    ds = pd.DataFrame(confusion, columns=class_names)
    dsc, ind = cluster_df(ds, k)
    ind = [x - 1 for x in ind]
    _, class_names_new = create_dict(ind, class_names)
    clustered_y = [ind[int(i)] for i in all_y]
    return clustered_y, class_names_new


def fit_predict(x, y, model, labels, name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # sc = StandardScaler()
    # x_train = sc.transform(x_train)
    # x_test = sc.transform(x_test)
    # y_train = sc.transform(y_train)
    # y_test = sc.transform(y_test)

    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
    predict_time = time.time()
    y_pred = model.predict(x_test)
    test_time = time.time() - predict_time

    no_classes = range(len(labels))
    fig, ax = plt.subplots()
    ax.set_ylim(bottom=0.14, top=0)
    plot_confusion_matrix(model, x_test, y_test, labels=no_classes,
                          display_labels=labels,
                          cmap='viridis',
                          include_values=False,
                          xticks_rotation='vertical',
                          normalize='pred',
                          ax=ax)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, labels=no_classes, average='macro')
    stats = '{0:_<20} CA: {1:5.3f} F1: {2:5.3f}'.format(name, accuracy, f1)
    # ax.set_title(stats)
    print(stats)

    save_figure(plt, name + '.png')
    return y_pred, y_test


k_best = ['INCLINATION', 'DEM', 'ARVI_max_mean_surf', 'ARVI_mean_val', 'NDVI_mean_val']
relief_f = ['DEM', 'BLUE_pos_len', 'GREEN_neg_len', 'RED_neg_len', 'BLUE_neg_len']
fastener_2k = ['DEM', 'ARVI_max_mean_len', 'BLUE_max_mean_surf', 'BLUE_mean_val', 'BLUE_neg_surf']
poss_2k = ['INCLINATION', 'DEM', 'ARVI_max_mean_surf', 'BLUE_neg_surf', 'NIR_pos_len']
fastener_600 = ['DEM', 'ARVI_max_mean_len', 'ARVI_max_mean_surf', 'BLUE_mean_val', 'GREEN_pos_surf']
poss_600 = ['INCLINATION', 'NDVI_max_mean_len', 'GREEN_mean_val', 'GREEN_neg_surf', 'RED_pos_len']

if __name__ == '__main__':
    x, y = get_data('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/genetic_samples2.csv', k_best)
    clf = tree.DecisionTreeClassifier()
    fit_predict(x, y, clf, class_names, 'SelectKBest')

    x, y = get_data('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/genetic_samples2.csv', relief_f)
    clf = tree.DecisionTreeClassifier()
    fit_predict(x, y, clf, class_names, 'reliefF')

    x, y = get_data('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/genetic_samples2.csv', fastener_2k)
    clf = tree.DecisionTreeClassifier()
    fit_predict(x, y, clf, class_names, 'FASTENER_2k')

    x, y = get_data('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/genetic_samples2.csv', poss_2k)
    clf = tree.DecisionTreeClassifier()
    fit_predict(x, y, clf, class_names, 'POSS_2k')

    x, y = get_data('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/genetic_samples2.csv', fastener_600)
    clf = tree.DecisionTreeClassifier()
    fit_predict(x, y, clf, class_names, 'FASTENER_600')

    x, y = get_data('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/genetic_samples2.csv', poss_600)
    clf = tree.DecisionTreeClassifier()
    fit_predict(x, y, clf, class_names, 'POSS_600')

    # print(x)
    # print(x[k_best])
    # DecisionTree
    # clf = tree.DecisionTreeClassifier()
    # fit_predict(x, y, clf, class_names, 'decision tree')

    # # LightGBM
    # lgb_model = lgb.LGBMClassifier(objective='multiclassova', num_class=len(class_names), metric='multi_logloss', )
    # y_pred, y_test = fit_predict(x, y, lgb_model, class_names, 'LGBM')
    # # clustered_y, class_names_new = form_clusters(y_pred, y_test, y, k=0.5)
    # # lgb_model = lgb.LGBMClassifier(objective='multiclassova', num_class=len(class_names_new), metric='multi_logloss')
    # # fit_predict(x, clustered_y, lgb_model, class_names_new, 'LGBM clustered')
    #
    # # DecisionTree
    # clf = tree.DecisionTreeClassifier()
    # y_pred, y_test = fit_predict(x, y, clf, class_names, 'decision tree')
    # # clustered_y, class_names_new = form_clusters(y_pred, y_test, y, k=0.6)
    # # fit_predict(x, clustered_y, clf, class_names_new, 'clustered tree')
    #
    # # Random Forest
    # rf_model = RandomForestClassifier()
    # y_pred, y_test = fit_predict(x, y, rf_model, class_names, 'random forest')
    # # clustered_y, class_names_new = form_clusters(y_pred, y_test, y, k=0.6)
    # # fit_predict(x, clustered_y, rf_model, class_names_new, 'clustered RF')
    #
    # # Logistic Regression
    # lr_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)
    # y_pred, y_test = fit_predict(x, y, lr_model, class_names, 'logistic regression')
    # # clustered_y, class_names_new = form_clusters(y_pred, y_test, y, k=0.6)
    # # fit_predict(x, clustered_y, lr_model, class_names_new, 'clustered logistic regression')
    #
    # # MLP
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 10, 10), random_state=1, max_iter=500)
    # y_pred, y_test = fit_predict(x, y, clf, class_names, 'MLP')
    # # clustered_y, class_names_new = form_clusters(y_pred, y_test, y, k=0.6)
    # # fit_predict(x, clustered_y, clf, class_names_new, 'clustered MLP')
