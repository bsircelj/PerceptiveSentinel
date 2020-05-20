from eolearn.core import EOPatch, FeatureType, OverwritePermission
import numpy as np
from sklearn.utils import resample
import random
import pandas as pd
import collections
import time
import datetime as dt
import random
from temporal_features import AddStreamTemporalFeaturesTask as st
from all_stream_features import allValid
import itertools as it
from os import path as ospath
from eolearn.core import EOTask, FeatureType
from scipy.ndimage import gaussian_filter
from eolearn.ml_tools.utilities import rolling_window

failed_pixels = 0


class AddStreamTemporalFeaturesTask():
    # pylint: disable=too-many-instance-attributes

    def __init__(self, data_feature=(FeatureType.DATA, 'NDVI'), data_index=None,
                 ndvi_feature_name=(FeatureType.DATA, 'NDVI'), mask_data=True, *,
                 max_val_feature='max_val', min_val_feature='min_val', mean_val_feature='mean_val',
                 sd_val_feature='sd_val', diff_max_feature='diff_max', diff_min_feature='diff_min',
                 diff_diff_feature='diff_diff', max_mean_feature='max_mean_feature',
                 max_mean_len_feature='max_mean_len', max_mean_surf_feature='max_mean_surf',
                 pos_surf_feature='pos_surf', pos_len_feature='pos_len', pos_rate_feature='pos_rate',
                 neg_surf_feature='neg_surf', neg_len_feature='neg_len', neg_rate_feature='neg_rate',
                 pos_transition_feature='pos_tran', neg_transition_feature='neg_tran',
                 feature_name_prefix=None, window_size=2, interval_tolerance=0.1, base_surface_min=-1.,
                 ndvi_barren_soil_cutoff=0.1):

        if feature_name_prefix:
            self.feature_name_prefix = feature_name_prefix
            if not feature_name_prefix.endswith("_"):
                self.feature_name_prefix += "_"
        else:
            self.feature_name_prefix = data_feature + "_"

        self.max_val_feature = self.feature_name_prefix + max_val_feature
        self.min_val_feature = self.feature_name_prefix + min_val_feature
        self.mean_val_feature = self.feature_name_prefix + mean_val_feature
        self.sd_val_feature = self.feature_name_prefix + sd_val_feature
        self.diff_max_feature = self.feature_name_prefix + diff_max_feature
        self.diff_min_feature = self.feature_name_prefix + diff_min_feature
        self.diff_diff_feature = self.feature_name_prefix + diff_diff_feature
        self.max_mean_feature = self.feature_name_prefix + max_mean_feature
        self.max_mean_len_feature = self.feature_name_prefix + max_mean_len_feature
        self.max_mean_surf_feature = self.feature_name_prefix + max_mean_surf_feature
        self.pos_surf_feature = self.feature_name_prefix + pos_surf_feature
        self.pos_len_feature = self.feature_name_prefix + pos_len_feature
        self.pos_rate_feature = self.feature_name_prefix + pos_rate_feature
        self.neg_surf_feature = self.feature_name_prefix + neg_surf_feature
        self.neg_len_feature = self.feature_name_prefix + neg_len_feature
        self.neg_rate_feature = self.feature_name_prefix + neg_rate_feature
        self.pos_transition_feature = self.feature_name_prefix + pos_transition_feature
        self.neg_transition_feature = self.feature_name_prefix + neg_transition_feature

        self.window_size = window_size
        self.interval_tolerance = interval_tolerance
        self.base_surface_min = base_surface_min

        self.ndvi_barren_soil_cutoff = ndvi_barren_soil_cutoff

    @staticmethod
    def derivative_features(mask, valid_dates, data, base_surface_min):
        """Calculates derivative based features for provided data points selected by
        mask (increasing data points, decreasing data points)

        :param mask: Mask indicating data points considered
        :type mask: np.array
        :param valid_dates: Dates (x-axis for surface calculation)
        :type valid_dates: np.array
        :param data: Base data
        :type data: np.array
        :param base_surface_min: Base surface value (added to each measurement)
        :type base_surface_min: float
        :return: Tuple of: maximal consecutive surface under the data curve,
                           date length corresponding to maximal surface interval,
                           rate of change in maximal interval,
                           (starting date index of maximal interval, ending date index of interval)
        """
        # index of 1 that have 0 before them, shifted by one to right
        up_mask = (mask[1:] == 1) & (mask[:-1] == 0)

        # Index of 1 that have 0 after them, correct indices
        down_mask = (mask[:-1] == 1) & (mask[1:] == 0)

        fst_der = np.where(up_mask[:-1])[0]
        snd_der = np.where(down_mask[1:])[0]
        der_ind_max = 0
        der_int_max = 0

        for ind, (start, end) in enumerate(zip(fst_der, snd_der)):

            integral = np.trapz(
                data[start:end + 1] - base_surface_min,
                valid_dates[start:end + 1])

            if abs(integral) >= abs(der_int_max):
                der_int_max = integral
                der_ind_max = ind
        try:
            start_ind = fst_der[der_ind_max]
            end_ind = snd_der[der_ind_max]
            der_len = valid_dates[end_ind] - valid_dates[start_ind]
            der_rate = (data[end_ind] - data[start_ind]) / der_len if der_len else 0
        except IndexError:
            der_len = der_rate = start_ind = end_ind = 0
            print("FAIL")
        return der_int_max, der_len, der_rate, (start_ind, end_ind)

    def execute(self, data, eopatch):
        """ Compute spatio-temporal features for input eopatch
        :param eopatch: Input eopatch
        :return: eopatch with computed spatio-temporal features
        """
        # pylint: disable=invalid-name
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        data = np.expand_dims(data, axis=-1)
        data = np.expand_dims(data, axis=-1)

        all_dates = np.asarray([x.toordinal() for x in eopatch.timestamp])

        valid_data_mask = np.ones_like(data)

        if data.ndim == 3:
            _, h, w = data.shape
        else:
            raise ValueError('{} feature has incorrect number of dimensions'.format(self.data_feature))

        madata = np.ma.array(data, dtype=np.float32, mask=~valid_data_mask.astype(np.bool))

        # Vectorized
        data_max_val = np.ma.MaskedArray.max(madata, axis=0)
        data_min_val = np.ma.MaskedArray.min(madata, axis=0)
        data_mean_val = np.ma.MaskedArray.mean(madata, axis=0)
        data_sd_val = np.ma.MaskedArray.std(madata, axis=0)

        data_diff_max = np.empty((h, w))
        data_diff_min = np.empty((h, w))
        # data_diff_diff = np.empty((h, w)) # Calculated later

        data_max_mean = np.empty((h, w))
        data_max_mean_len = np.empty((h, w))
        data_max_mean_surf = np.empty((h, w))

        data_pos_surf = np.empty((h, w))
        data_pos_len = np.empty((h, w))
        data_pos_rate = np.empty((h, w))

        data_neg_surf = np.empty((h, w))
        data_neg_len = np.empty((h, w))
        data_neg_rate = np.empty((h, w))

        data_pos_tr = np.empty((h, w))
        data_neg_tr = np.empty((h, w))
        for ih, iw in it.product(range(h), range(w)):
            data_curve = madata[:, ih, iw]
            valid_idx = np.where(~madata.mask[:, ih, iw])[0]

            data_curve = data_curve[valid_idx].filled()

            valid_dates = all_dates[valid_idx]

            sw_max = np.max(rolling_window(data_curve, self.window_size), -1)
            sw_min = np.min(rolling_window(data_curve, self.window_size), -1)

            sw_diff = sw_max - sw_min

            data_diff_max[ih, iw] = np.max(sw_diff)
            data_diff_min[ih, iw] = np.min(sw_diff)

            sw_mean = np.mean(rolling_window(data_curve, self.window_size), -1)
            max_mean = np.max(sw_mean)

            data_max_mean[ih, iw] = max_mean

            # Calculate max mean interval
            # Work with mean windowed or whole set?
            workset = data_curve  # or sw_mean, which is a bit more smoothed
            higher_mask = workset >= max_mean - ((1 - self.interval_tolerance) * abs(max_mean))

            # Just normalize to have 0 on each side
            higher_mask_norm = np.zeros(len(higher_mask) + 2)
            higher_mask_norm[1:len(higher_mask) + 1] = higher_mask

            # index of 1 that have 0 before them, SHIFTED BY ONE TO RIGHT
            up_mask = (higher_mask_norm[1:] == 1) & (higher_mask_norm[:-1] == 0)

            # Index of 1 that have 0 after them, correct indices
            down_mask = (higher_mask_norm[:-1] == 1) & (higher_mask_norm[1:] == 0)

            # Calculate length of interval as difference between times of first and last high enough observation,
            # in particular, if only one such observation is high enough, the length of such interval is 0
            # One can extend this to many more ways of calculating such length:
            # take forward/backward time differences, interpolate in between (again...) and treat this as
            # continuous problem, take mean of the time intervals between borders...
            times_up = valid_dates[up_mask[:-1]]
            times_down = valid_dates[down_mask[1:]]

            # There may be several such intervals, take the longest one
            times_diff = times_down - times_up
            # if there are no such intervals, the signal is constant,
            # set everything to zero and continue
            if times_diff.size == 0:
                data_max_mean_len[ih, iw] = 0
                data_max_mean_surf[ih, iw] = 0

                data_pos_surf[ih, iw] = 0
                data_pos_len[ih, iw] = 0
                data_pos_rate[ih, iw] = 0

                data_neg_surf[ih, iw] = 0
                data_neg_len[ih, iw] = 0
                data_neg_rate[ih, iw] = 0

            max_ind = np.argmax(times_diff)
            data_max_mean_len[ih, iw] = times_diff[max_ind]

            fst = np.where(up_mask[:-1])[0]
            snd = np.where(down_mask[1:])[0]

            surface = np.trapz(data_curve[fst[max_ind]:snd[max_ind] + 1] - self.base_surface_min,
                               valid_dates[fst[max_ind]:snd[max_ind] + 1])
            data_max_mean_surf[ih, iw] = surface

            # Derivative based features
            # How to approximate derivative?
            derivatives = np.gradient(data_curve, valid_dates)

            # Positive derivative
            pos = np.zeros(len(derivatives) + 2)
            pos[1:len(derivatives) + 1] = derivatives >= 0

            pos_der_int, pos_der_len, pos_der_rate, (start, _) = \
                self.derivative_features(pos, valid_dates, data_curve, self.base_surface_min)

            data_pos_surf[ih, iw] = pos_der_int
            data_pos_len[ih, iw] = pos_der_len
            data_pos_rate[ih, iw] = pos_der_rate

            neg = np.zeros(len(derivatives) + 2)
            neg[1:len(derivatives) + 1] = derivatives <= 0

            neg_der_int, neg_der_len, neg_der_rate, (_, end) = \
                self.derivative_features(neg, valid_dates, data_curve, self.base_surface_min)

            data_neg_surf[ih, iw] = neg_der_int
            data_neg_len[ih, iw] = neg_der_len
            data_neg_rate[ih, iw] = neg_der_rate

        df = dict()
        df[self.max_val_feature] = float(data_max_val.squeeze())
        df[self.min_val_feature] = float(data_min_val.squeeze())
        df[self.mean_val_feature] = float(data_mean_val.squeeze())
        df[self.sd_val_feature] = float(data_sd_val.squeeze())

        df[self.diff_max_feature] = float(data_diff_max.squeeze())
        df[self.diff_min_feature] = float(data_diff_min.squeeze())
        df[self.diff_diff_feature] = float((data_diff_max - data_diff_min).squeeze())

        df[self.max_mean_feature] = float(data_max_mean.squeeze())
        df[self.max_mean_len_feature] = float(data_max_mean_len.squeeze())
        df[self.max_mean_surf_feature] = float(data_max_mean_surf.squeeze())

        df[self.pos_len_feature] = float(data_pos_len.squeeze())
        df[self.pos_surf_feature] = float(data_pos_surf.squeeze())
        df[self.pos_rate_feature] = float(data_pos_rate.squeeze())
        df[self.pos_transition_feature] = float(data_pos_tr.squeeze())

        df[self.neg_len_feature] = float(data_neg_len.squeeze())
        df[self.neg_surf_feature] = float(data_neg_surf.squeeze())
        df[self.neg_rate_feature] = float(data_neg_rate.squeeze())
        df[self.neg_transition_feature] = float(data_neg_tr.squeeze())
        return df


class AddBaseFeatures:

    def __init__(self, c1=6, c2=7.5, L=1):
        self.c1 = c1
        self.c2 = c2
        self.L = L

    def execute(self, nir, blue, green, red):
        # nir = eopatch.data['BANDS'][..., [7]]
        # blue = eopatch.data['BANDS'][..., [1]]
        # green = eopatch.data['BANDS'][..., [2]]
        # red = eopatch.data['BANDS'][..., [3]]

        arvi = np.clip((nir - (2 * red) + blue) / (nir + (2 * red) + blue + 0.000000001), -1, 1)

        evi = np.clip(2.5 * ((nir - red) / (nir + (self.c1 * red) - (self.c2 * blue) + self.L + 0.000000001)), -1, 1)
        ndvi = np.clip((nir - red) / (nir + red + 0.000000001), -1, 1)

        ndwi = np.clip((green - nir) / (green + nir + 0.000000001), -1, 1)

        sipi = np.clip((nir - blue) / (nir - red + 0.000000001), 0, 2)

        Lvar = 0.5
        savi = np.clip(((nir - red) / (nir + red + Lvar + 0.000000001)) * (1 + Lvar), -1, 1)

        return arvi, evi, ndvi, ndwi, sipi, savi


if __name__ == '__main__':
    # samples_path = '/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/enriched_samples9797.csv'  # CHANGE

    print('start ' + str(dt.datetime.now()))
    # samples_path = '/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/'  # CHANGE
    samples_path = 'D:\\Samples\\'
    patches_path = 'E:\\Data\\PerceptiveSentinel\\Slovenia'
    # patches_path = '/home/beno/Documents/test/Slovenia'

    patches_path_s1 = 'E:\\Data\\PerceptiveSentinel\\Slovenia_S1'

    dataset = pd.read_csv(samples_path + 'enriched_samples9797.csv')
    dataset.drop(columns=['Unnamed: 0', 'NDVI_sd_val', 'EVI_min_val', 'ARVI_max_mean_len', 'SIPI_mean_val',
                          'NDVI_min_val', 'SAVI_min_val'], inplace=True)
    dataset.sort_values(by='patch_no', inplace=True)
    no = dataset.shape[0]
    # no=10
    base_names = ['ARVI', 'EVI', 'NDVI', 'NDWI', 'SIPI', 'SAVI', 'BLUE', 'GREEN', 'RED', 'NIR']

    suffix_name = ['_diff_diff', '_diff_max', '_diff_min', '_max_mean_feature', '_max_mean_len',
                   '_max_mean_surf',
                   '_max_val', '_mean_val', '_min_val', '_neg_len', '_neg_rate', '_neg_surf', '_neg_tran',
                   '_pos_len', '_pos_rate', '_pos_surf', '_pos_tran', '_sd_val']
    columns = dataset.columns
    dataset = dataset.to_dict(orient='list')
    dataset['DEM'] = np.zeros(no)

    # columns = np.concatenate(columns,['DEM'])
    # placeholder = np.zeros(dataset.shape[0])
    for b in base_names:
        for s in suffix_name:
            dataset[b + s] = np.zeros(no)

    base_names_s1 = ['VV', 'VH', 'VV_spring', 'VV_summer', 'VV_autumn', 'VV_winter', 'VH_spring', 'VH_summer',
                     'VH_autumn', 'VH_winter']
    suffix_name_s1 = ['_avg', '_max', '_min', '_std']

    for b in base_names_s1:
        for s in suffix_name_s1:
            dataset[b + s] = np.zeros(no)

    color_names = [[1, 'BLUE'],
                   [3, 'GREEN'],
                   [3, 'RED'],
                   [7, 'NIR']]

    color_stream = [AddStreamTemporalFeaturesTask(data_feature=c) for _, c in color_names]
    index_stream = [AddStreamTemporalFeaturesTask(data_feature=c) for c in base_names[0:6]]

    # eopatch = EOPatch.load('{}/eopatch_{}'.format(patches_path, 0), lazy_loading=True)
    # t, _, _, _ = eopatch.data['BANDS'].shape
    bands = np.zeros(no)
    base_features = AddBaseFeatures()
    update = False
    update_per = 1
    starttime = time.time()
    patch_id = -1
    eopatch = EOPatch()
    eopatch_s1 = EOPatch()
    for x in range(no):
    # for x in range(1):
        # percent = x / no
        # if percent > update_per:
        #     update = True
        #     update_per += 0.5
        # if update:
        #     update = False
        # elapsed = time.time() - starttime
        # end = (no - x) * (elapsed / (x + 0.001))
        # print('{0:%} time {1:10.2f} remaining {2:.2f}h'.format(percent, elapsed, end / 3600))
        patch_id_new = dataset['patch_no'][x]
        w = int(dataset['x'][x])
        h = int(dataset['y'][x])
        # w = np.clip(w, 0, 300)
        # h = np.clip(h, 0, 300)
        p1 = '{}/eopatch_{}'.format(patches_path, int(patch_id))
        p2 = '{}/eopatch_{}'.format(patches_path_s1, int(patch_id))
        if not ospath.exists(p1) or not ospath.exists(p2):
            continue

        if patch_id != patch_id_new:
            patch_id = patch_id_new
            eopatch = EOPatch.load(p1, lazy_loading=True)  # CHANGE
            eopatch_s1 = EOPatch.load(p2, lazy_loading=True)
        dataset['DEM'][x] = eopatch.data_timeless['DEM'][h][w].squeeze()
        # dataset.at['DEM', x] = eopatch.data_timeless['DEM'][h][w].squeeze()
        # dataset.set_value('DEM',x,eopatch.data_timeless['DEM'][h][w].squeeze() )
        # print(dataset.at['DEM', x])
        ti, _, _, _ = eopatch.data['BANDS'].shape
        # one_pixel = np.zeros((len(color_names), ti))
        si = 0
        for c_index, c_name in color_names:
            one_pixel = np.zeros(ti)
            for t in range(ti):
                one_pixel[t] = eopatch.data['BANDS'][t][h][w][c_index]
            # print(one_pixel)
            try:
                pix_features = color_stream[si].execute(one_pixel, eopatch)
            except:
                failed_pixels += 1
                continue
            si += 1
            for keys in pix_features:
                dataset[keys][x] = pix_features[keys]

        indices = np.zeros((ti, 6))
        for t in range(ti):
            nir = eopatch.data['BANDS'][t][h][w][7]
            blue = eopatch.data['BANDS'][t][h][w][1]
            green = eopatch.data['BANDS'][t][h][w][2]
            red = eopatch.data['BANDS'][t][h][w][3]
            indices[t] = base_features.execute(nir, blue, green, red)

        for i in range(6):
            pix_features = index_stream[i].execute(indices[:, i], eopatch)
            for keys in pix_features:
                dataset[keys][x] = pix_features[keys]

        # S1 features
        vv = eopatch_s1.data['IW'][:][h][w][0]
        vh = eopatch_s1.data['IW'][:][h][w][1]
        vv_sig5 = gaussian_filter(vv, sigma=5)
        vh_sig5 = gaussian_filter(vh, sigma=5)
        winter, spring, summer, autumn = ([], [], [], [])
        winter_vh, spring_vh, summer_vh, autumn_vh = ([], [], [], [])

        for i, dt0 in enumerate(eopatch_s1.timestamp):
            m = dt0.month
            if 6 > m >= 3:
                spring += [vv_sig5[i]]
                spring_vh += [vh_sig5[i]]
            elif 9 > m >= 6:
                summer += [vv_sig5[i]]
                summer_vh += [vh_sig5[i]]
            elif 11 > m >= 9:
                autumn += [vv_sig5]
                autumn_vh += [vh_sig5[i]]
            else:
                winter += [vv_sig5[i]]
                winter_vh += [vh_sig5[i]]

        name_and = [('VV', vv_sig5),
                    ('VH', vh_sig5),
                    ('VV_spring', spring),
                    ('VV_summer', summer),
                    ('VV_autumn', autumn),
                    ('VV_winter', winter),
                    ('VH_spring', spring_vh),
                    ('VH_summer', summer, vh),
                    ('VH_autumn', autumn_vh),
                    ('VH_winter', winter_vh)]

        for name, data in name_and:
            dataset[name + '_avg'] = np.average(data)
            dataset[name + '_max'] = np.amax(data)
            dataset[name + '_min'] = np.amin(data)
            dataset[name + '_std'] = np.std(data)

    # print("Done. only {} failed".format(failed_pixels))
    filename = 'extended_samples' + str(int(random.random() * 10000))
    # print(dataset)
    dataset = pd.DataFrame.from_dict(dataset)
    # print(dataset)
    dataset.to_csv(samples_path + filename + '.csv', index=False)
