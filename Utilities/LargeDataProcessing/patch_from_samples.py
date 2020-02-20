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
from all_stream_features import AddBaseFeatures, allValid
import itertools as it

import numpy as np

from eolearn.core import EOTask, FeatureType

from eolearn.ml_tools.utilities import rolling_window


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
        der_ind_max = -1
        der_int_max = -1

        for ind, (start, end) in enumerate(zip(fst_der, snd_der)):

            integral = np.trapz(
                data[start:end + 1] - base_surface_min,
                valid_dates[start:end + 1])

            if abs(integral) >= abs(der_int_max):
                der_int_max = integral
                der_ind_max = ind

        start_ind = fst_der[der_ind_max]
        end_ind = snd_der[der_ind_max]

        der_len = valid_dates[end_ind] - valid_dates[start_ind]
        der_rate = (data[end_ind] - data[start_ind]) / der_len if der_len else 0

        return der_int_max, der_len, der_rate, (start_ind, end_ind)

    def execute(self, data, eopatch):
        """ Compute spatio-temporal features for input eopatch
        :param eopatch: Input eopatch
        :return: eopatch with computed spatio-temporal features
        """
        # pylint: disable=invalid-name
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        all_dates = np.asarray([x.toordinal() for x in eopatch.timestamp])

        valid_data_mask = np.ones_like(data)

        if data.ndim == 3:
            _, h, w = data.shape
        else:
            raise ValueError(' feature has incorrect number of dimensions')

        madata = np.ma.array(data, dtype=np.float32, mask=~valid_data_mask.astype(np.bool))

        # Vectorized
        data_max_val = np.ma.MaskedArray.max(madata, axis=0).filled()
        data_min_val = np.ma.MaskedArray.min(madata, axis=0).filled()
        data_mean_val = np.ma.MaskedArray.mean(madata, axis=0).filled()
        data_sd_val = np.ma.MaskedArray.std(madata, axis=0).filled()

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

        df = pd.DataFrame()
        df[self.max_val_feature] = data_max_val
        df[self.min_val_feature] = data_min_val
        df[self.mean_val_feature] = data_mean_val
        df[self.sd_val_feature] = data_sd_val

        df[self.diff_max_feature] = data_diff_max
        df[self.diff_min_feature] = data_diff_min
        df[self.diff_diff_feature] = (data_diff_max - data_diff_min)

        df[self.max_mean_feature] = data_max_mean
        df[self.max_mean_len_feature] = data_max_mean_len
        df[self.max_mean_surf_feature] = data_max_mean_surf

        df[self.pos_len_feature] = data_pos_len
        df[self.pos_surf_feature] = data_pos_surf
        df[self.pos_rate_feature] = data_pos_rate
        df[self.pos_transition_feature] = data_pos_tr

        df[self.neg_len_feature] = data_neg_len
        df[self.neg_surf_feature] = data_neg_surf
        df[self.neg_rate_feature] = data_neg_rate
        df[self.neg_transition_feature] = data_neg_tr

        return df


if __name__ == '__main__':
    samples_path = '/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/enriched_samples9797.csv'  # CHANGE

    # patches_path = 'E:/Data/PerceptiveSentinel/Slovenia'
    patches_path = '/home/beno/Documents/test/Slovenia'

    dataset = pd.read_csv(samples_path)
    no = dataset.shape[0]

    color_names = [[1, 'BLUE'],
                  [3, 'GREEN'],
                  [3, 'RED'],
                  [7, 'NIR']]

    for sample in range(no):  # CHANGE
        patch_id = dataset['patch_no'][sample]
        w = dataset['x'][sample]
        h = dataset['y'][sample]
        w = np.clip(w, 0, 300)
        h = np.clip(h, 0, 300)

        eopatch = EOPatch.load('{}/eopatch_{}'.format(patches_path, 0), lazy_loading=True)
        ti, _, _, _ = eopatch.data['BANDS'].shape
        bands = np.zeros(ti)

        for t in range(ti):
            bands[t] = eopatch.data['BANDS'][t][h][w][]

    new_patch = EOPatch()
    new_patch.add_feature(FeatureType.DATA, 'BANDS', bands)
    for sample in range(13):
        one_band = bands[..., sample]
        one_band = np.expand_dims(one_band, axis=-1)
        new_patch.add_feature(FeatureType.DATA, 'B{}'.format(int(sample + 1)), one_band)

    new_patch.timestamp = EOPatch.load('{}/eopatch_{}'.format(patches_path, 0), lazy_loading=True).timestamp  # CHANGE
    # new_patch.add_feature(FeatureType.MASK, 'VALID_DATA', np.ones((69, 1, no, 1)))
    new_patch = allValid('VALID_DATA').execute(new_patch)

    new_patch = AddBaseFeatures().execute(new_patch)
    features = ['NDVI', 'SAVI', 'SIPI', 'EVI', 'ARVI', 'NDWI']
    for f in features:
        print(f + '\n')
        new_patch = st(data_feature=f).execute(new_patch)

    print('BLUE\n')
    new_patch = st(data_feature='B2', feature_name_prefix='BLUE', ).execute(new_patch)
    print('GREEN\n')
    new_patch = st(data_feature='B3', feature_name_prefix='GREEN').execute(new_patch)
    print('RED\n')
    new_patch = st(data_feature='B4', feature_name_prefix='RED').execute(new_patch)
    print('NIR\n')
    new_patch = st(data_feature='B8', feature_name_prefix='NIR').execute(new_patch)

    print(new_patch)
    all_features = new_patch.get_feature_list()
    all_data_timeless = []
    for f in all_features:
        if type(f) is tuple:
            all_data_timeless.append((f[0], f[1]))

    for f in all_data_timeless:
        dataset[f] = new_patch.data_timeless[f].squeeze()

    filename = 'extended_samples' + str(int(random.random() * 10000))

    dataset.to_csv('/home/beno/Documents/IJS/Perceptive-Sentinel/Samples/' + filename + '.csv')
