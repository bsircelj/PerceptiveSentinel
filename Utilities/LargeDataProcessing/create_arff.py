import numpy as np
import pandas as pd

crop_names = {0: 'Not Farmland', 1: 'Beans', 2: 'Beets', 3: 'Buckwheat', 4: 'Fallow land', 5: 'Grass', 6: 'Hop',
              7: 'Legumes or grass', 8: 'Maize', 9: 'Meadows', 10: 'Orchards', 11: 'Other', 12: 'Peas', 13: 'Poppy',
              14: 'Potatoes', 15: 'Pumpkins', 16: 'Soft fruits', 17: 'Soybean', 18: 'Summer cereals', 19: 'Sun flower',
              20: 'Vegetables', 21: 'Vineyards', 22: 'Winter cereals', 23: 'Winter rape'}

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

feature_subset = ['ARVI_max_mean_len']

path = '/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/'


# path = 'D:\\Samples\\'

# def create_header():


def create_arff(name, samples_names, feature_names):
    dataset = pd.read_csv(path + samples_names)

    y = dataset['LPIS_2017'].to_numpy()
    # !!!! -1 is marking no LPIS data so everything is shifted by one cause some classifiers don't want negative numbers
    y = [int(a + 1) for a in y]
    x = dataset[feature_names].to_numpy()

    all_class_names = [crop_names[int(rename)].replace(' ', '_') for rename in y]
    # other = [all_class_names, y, clustered_y, clustered_y2]

    x_all = np.concatenate((np.array(all_class_names)[:, np.newaxis],
                            # np.array(y)[:, np.newaxis],
                            np.array(x)),
                           axis=1)

    df = pd.DataFrame(x_all)

    file = open(path + '{}.csv'.format(name), 'a')
    #
    # file.write(
    #     '% Collection of land patch samples\n% 	Location: Slovenija\n% 	Year: 2017\n\n% 	Number of samples per class: 20 000\n% 	Number of classes: 24\n\n@RELATION LPIS_Slovenia_2017\n\n')
    # file.write('% Citation:\n')
    # file.write('% Project: Perceptive Sentinel, H2020\n')
    # file.write(
    #     '% Parameters: The EO data were collected for the whole year.\4 raw band measurements (red, green, blue - RGB and near infrared - NIR) and 6 relevant vegetation-\ '
    #     'related derived indices (normalized differential vegetation index - NDVI, normalized differential \
    #     water index - NDWI, enhanced vegetation index - EVI, soil-adjusted vegetation index - SAVI, structure\
    #      intensive pigment index - SIPI and atmospherically resistant vegetation index - ARVI).\
    #      The derived indices are based on extensive domain knowledge and are used for assessing vegetation properties. ')
    #
    # file.write(
    #     '@ATTRIBUTE {:25} {Not_Farmland, Beans, Beets, Buckwheat, Fallow_land, Grass, Hop, Legumes_or_grass, Maize, Meadows, Orchards, Other, Peas, Poppy, Potatoes, Pumpkins, Soft_fruits, Soybean, Summer_cereals, Sun_flower, Vegetables, Vineyards, Winter_cereals, Winter_rape}'.format(
    #         'LPIS_2017'))

    # columns = np.concatenate((['class_name'], feature_names), axis=0)
    for c in feature_names:
        file.write('@ATTRIBUTE {:25} NUMERIC\n'.format(c))

    file.write('\n@DATA\n')
    df.to_csv(file, header=False, index=False)
    file.close()


if __name__ == '__main__':
    # fet = all_features
    fet = feature_subset
    create_arff('FASTENER_dataset', 'enriched_samples10000.csv', fet)
