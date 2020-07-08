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
indexes = [1, 6, 7, 115, 117, 121, 143]

i1 = [1, 6, 7, 117, 143]
i2 = [0, 1, 7, 9, 45]
i3 = [1, 123, 137, 155, 119, ]
i4 = [1, 6, 115, 117, 121]
i5 = [0, 1, 7, 121, 177]
i6 = [1, 6, 7, 117, 143]
i7 = [0, 42, 135, 139, 159]

all_relevant = i1
for i in [i1, i2, i3, i4, i5, i6, i7]:
    all_relevant = np.union1d(all_relevant, i)


def p(lst):
    all = []
    for i in lst:
        all = np.concatenate((all, [all_features[i]]))
    print(all)

if __name__ == '__main__':
    print(all_relevant)
    p(i2)
    p(i3)
    p(i4)
    p(i5)
    p(i6)
    p(i7)
    # for i in i1:
    #     print(all_features[i])
    # all = []
    # for i in all_relevant:
    #     all = np.concatenate((all, [all_features[i]]))
    #
    # all = np.concatenate((all, ['LPIS_2017', 'patch_no', 'x', 'y']))
    #
    # dataset = pd.read_csv('D:\\Samples\\extended_samples9953.csv')
    # # print(dataset.columns)
    # new_dat = dataset[all]
    # # print(new_dat)
    # new_dat.to_csv('D:/Samples/genetic_samples2.csv')
