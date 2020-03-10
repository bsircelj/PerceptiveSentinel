import numpy as np
import pandas as pd


all_features = ['INCLINATION', 'DEM', 'ARVI_diff_diff', 'ARVI_diff_max', 'ARVI_diff_min'
    , 'ARVI_max_mean_feature', 'ARVI_max_mean_len', 'ARVI_max_mean_surf'
    , 'ARVI_max_val', 'ARVI_mean_val', 'ARVI_min_val', 'ARVI_neg_len'
    , 'ARVI_neg_rate', 'ARVI_neg_surf', 'ARVI_neg_tran', 'ARVI_pos_len'
    , 'ARVI_pos_rate', 'ARVI_pos_surf', 'ARVI_pos_tran', 'ARVI_sd_val'
    , 'EVI_diff_diff', 'EVI_diff_max', 'EVI_diff_min', 'EVI_max_mean_feature'
    , 'EVI_max_mean_len', 'EVI_max_mean_surf', 'EVI_max_val', 'EVI_mean_val'
    , 'EVI_min_val', 'EVI_neg_len', 'EVI_neg_rate', 'EVI_neg_surf', 'EVI_neg_tran'
    , 'EVI_pos_len', 'EVI_pos_rate', 'EVI_pos_surf', 'EVI_pos_tran', 'EVI_sd_val'
    , 'NDVI_diff_diff', 'NDVI_diff_max', 'NDVI_diff_min', 'NDVI_max_mean_feature'
    , 'NDVI_max_mean_len', 'NDVI_max_mean_surf', 'NDVI_max_val', 'NDVI_mean_val'
    , 'NDVI_min_val', 'NDVI_neg_len', 'NDVI_neg_rate', 'NDVI_neg_surf'
    , 'NDVI_neg_tran', 'NDVI_pos_len', 'NDVI_pos_rate', 'NDVI_pos_surf'
    , 'NDVI_pos_tran', 'NDVI_sd_val', 'NDWI_diff_diff', 'NDWI_diff_max'
    , 'NDWI_diff_min', 'NDWI_max_mean_feature', 'NDWI_max_mean_len'
    , 'NDWI_max_mean_surf', 'NDWI_max_val', 'NDWI_mean_val', 'NDWI_min_val'
    , 'NDWI_neg_len', 'NDWI_neg_rate', 'NDWI_neg_surf', 'NDWI_neg_tran'
    , 'NDWI_pos_len', 'NDWI_pos_rate', 'NDWI_pos_surf', 'NDWI_pos_tran'
    , 'NDWI_sd_val', 'SIPI_diff_diff', 'SIPI_diff_max', 'SIPI_diff_min'
    , 'SIPI_max_mean_feature', 'SIPI_max_mean_len', 'SIPI_max_mean_surf'
    , 'SIPI_max_val', 'SIPI_mean_val', 'SIPI_min_val', 'SIPI_neg_len'
    , 'SIPI_neg_rate', 'SIPI_neg_surf', 'SIPI_neg_tran', 'SIPI_pos_len'
    , 'SIPI_pos_rate', 'SIPI_pos_surf', 'SIPI_pos_tran', 'SIPI_sd_val'
    , 'SAVI_diff_diff', 'SAVI_diff_max', 'SAVI_diff_min', 'SAVI_max_mean_feature'
    , 'SAVI_max_mean_len', 'SAVI_max_mean_surf', 'SAVI_max_val', 'SAVI_mean_val'
    , 'SAVI_min_val', 'SAVI_neg_len', 'SAVI_neg_rate', 'SAVI_neg_surf'
    , 'SAVI_neg_tran', 'SAVI_pos_len', 'SAVI_pos_rate', 'SAVI_pos_surf'
    , 'SAVI_pos_tran', 'SAVI_sd_val', 'BLUE_diff_diff', 'BLUE_diff_max'
    , 'BLUE_diff_min', 'BLUE_max_mean_feature', 'BLUE_max_mean_len'
    , 'BLUE_max_mean_surf', 'BLUE_max_val', 'BLUE_mean_val', 'BLUE_min_val'
    , 'BLUE_neg_len', 'BLUE_neg_rate', 'BLUE_neg_surf', 'BLUE_neg_tran'
    , 'BLUE_pos_len', 'BLUE_pos_rate', 'BLUE_pos_surf', 'BLUE_pos_tran'
    , 'BLUE_sd_val', 'GREEN_diff_diff', 'GREEN_diff_max', 'GREEN_diff_min'
    , 'GREEN_max_mean_feature', 'GREEN_max_mean_len', 'GREEN_max_mean_surf'
    , 'GREEN_max_val', 'GREEN_mean_val', 'GREEN_min_val', 'GREEN_neg_len'
    , 'GREEN_neg_rate', 'GREEN_neg_surf', 'GREEN_neg_tran', 'GREEN_pos_len'
    , 'GREEN_pos_rate', 'GREEN_pos_surf', 'GREEN_pos_tran', 'GREEN_sd_val'
    , 'RED_diff_diff', 'RED_diff_max', 'RED_diff_min', 'RED_max_mean_feature'
    , 'RED_max_mean_len', 'RED_max_mean_surf', 'RED_max_val', 'RED_mean_val'
    , 'RED_min_val', 'RED_neg_len', 'RED_neg_rate', 'RED_neg_surf', 'RED_neg_tran'
    , 'RED_pos_len', 'RED_pos_rate', 'RED_pos_surf', 'RED_pos_tran', 'RED_sd_val'
    , 'NIR_diff_diff', 'NIR_diff_max', 'NIR_diff_min', 'NIR_max_mean_feature'
    , 'NIR_max_mean_len', 'NIR_max_mean_surf', 'NIR_max_val', 'NIR_mean_val'
    , 'NIR_min_val', 'NIR_neg_len', 'NIR_neg_rate', 'NIR_neg_surf', 'NIR_neg_tran'
    , 'NIR_pos_len', 'NIR_pos_rate', 'NIR_pos_surf', 'NIR_pos_tran', 'NIR_sd_val']

def create_header():


def create_arff(model, name,feature_names):
    x, y = get_data('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/enriched_samples9797.csv', feature_names)
    # lgb_model = lgb.LGBMClassifier(objective='multiclassova', num_class=len(class_names), metric='multi_logloss', )
    y_pred, y_test, x_test = fit_predict(x, y, model, class_names, 'LGBM')
    clustered_y, class_names_new = form_clusters(y_pred, y_test, y, k=0.5)
    lgb_model1 = lgb.LGBMClassifier(objective='multiclassova', num_class=len(class_names_new), metric='multi_logloss')
    fit_predict(x, clustered_y, lgb_model1, class_names_new, 'LGBM_partially_aggerated')
    for c in enumerate(class_names_new):
        print('% {0:2}: {1}', format(c))

    clustered_y2, class_names_new2 = form_clusters(y_pred, y_test, y, k=0.6)
    lgb_model2 = lgb.LGBMClassifier(objective='multiclassova', num_class=len(class_names_new), metric='multi_logloss')
    fit_predict(x, clustered_y2, lgb_model2, class_names_new2, 'LGBM_fully_aggregated')
    for c in enumerate(class_names_new2):
        print('% {0:2}: {1}', format(c))

    all_class_names = [class_names[int(rename)] for rename in y]
    # other = [all_class_names, y, clustered_y, clustered_y2]

    x_all = np.concatenate((x,
                            np.array(all_class_names)[:, np.newaxis],
                            np.array(y)[:, np.newaxis],
                            np.array(clustered_y)[:, np.newaxis],
                            np.array(clustered_y2)[:, np.newaxis]),
                           axis=1)
    acc = ''
    for c in class_names:
        acc = acc + c + ','
    print(acc)
    df = pd.DataFrame(x_all)

    file = open('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/{}.csv'.format(name), 'a')
    columns = np.concatenate((feature_names, ['class_name', 'raw', 'partially_aggregated', 'fully_aggregated']), axis=0)

    for c in columns:
        file.write('@ATTRIBUTE {:25} NUMERIC\n'.format(c))

    df.to_csv(file, header=False, index=False)
    file.close()
