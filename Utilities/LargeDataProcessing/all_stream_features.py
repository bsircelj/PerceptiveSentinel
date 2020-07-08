from eolearn.ml_tools.utilities import rolling_window
from scipy import ndimage
import enum
import numpy as np
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk

import warnings

warnings.filterwarnings("ignore")
from temporal_features import AddStreamTemporalFeaturesTask
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector, AddValidDataMaskTask

from eolearn.core import EOTask, EOPatch, LinearWorkflow, EOWorkflow, Dependency, FeatureType, OverwritePermission, \
    LoadTask, SaveTask, EOExecutor
import os
from eolearn.features import LinearInterpolation, SimpleFilterTask

from eolearn.io import SentinelHubDemTask


class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """

    def __call__(self, eopatch):
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))


class AddGradientTask(EOTask):
    def __init__(self, elevation_feature, result_feature):
        self.feature = elevation_feature
        self.result_feature = result_feature

    def execute(self, eopatch):
        elevation = eopatch[self.feature[0]][self.feature[1]].squeeze()
        gradient = ndimage.gaussian_gradient_magnitude(elevation, 1)
        eopatch.add_feature(self.result_feature[0], self.result_feature[1], gradient[..., np.newaxis])

        return eopatch


class printPatch(EOTask):
    def __init__(self, message="\npatch:"):
        self.message = message

    def execute(self, eopatch):
        eopatch.meta_info['service_type'] = 'wms'
        print(self.message)
        print(eopatch)
        return eopatch


class allValid(EOTask):

    def __init__(self, mask_name):
        self.mask_name = mask_name

    def execute(self, eopatch):
        # print(eopatch)
        t, w, h, _ = eopatch.data['BANDS'].shape
        eopatch.add_feature(FeatureType.MASK, self.mask_name, np.ones((t, w, h, 1)))
        return eopatch


class LULC(enum.Enum):
    NO_DATA = (0, 'No Data', 'white')
    CULTIVATED_LAND = (1, 'Cultivated Land', 'xkcd:lime')
    FOREST = (2, 'Forest', 'xkcd:darkgreen')
    GRASSLAND = (3, 'Grassland', 'orange')
    SHRUBLAND = (4, 'Shrubland', 'xkcd:tan')
    WATER = (5, 'Water', 'xkcd:azure')
    WETLAND = (6, 'Wetlands', 'xkcd:lightblue')
    TUNDRA = (7, 'Tundra', 'xkcd:lavender')
    ARTIFICIAL_SURFACE = (8, 'Artificial Surface', 'crimson')
    BARELAND = (9, 'Bareland', 'xkcd:beige')
    SNOW_AND_ICE = (10, 'Snow and Ice', 'black')

    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3


def normalize_feature(feature):  # Assumes similar max and min throughout different features
    f_min = np.min(feature)
    f_max = np.max(feature)
    if f_max != 0:
        return (feature - f_min) / (f_max - f_min)


def temporal_derivative(data, window_size=(3,)):
    padded_slope = np.zeros(data.shape)
    window = rolling_window(data, window_size, axes=0)

    slope = window[..., -1] - window[..., 0]  # TODO Missing division with time
    padded_slope[1:-1] = slope  # Padding with zeroes at the beginning and end

    return normalize_feature(padded_slope)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class AddBaseFeatures(EOTask):

    def __init__(self, c1=6, c2=7.5, L=1):
        self.c1 = c1
        self.c2 = c2
        self.L = L

    def execute(self, eopatch):
        nir = eopatch.data['BANDS'][..., [7]]
        blue = eopatch.data['BANDS'][..., [1]]
        red = eopatch.data['BANDS'][..., [3]]
        green = eopatch.data['BANDS'][..., [2]]

        eopatch.add_feature(FeatureType.DATA, 'NIR', nir)
        eopatch.add_feature(FeatureType.DATA, 'BLUE', blue)
        eopatch.add_feature(FeatureType.DATA, 'GREEN', green)

        arvi = np.clip((nir - (2 * red) + blue) / (nir + (2 * red) + blue + 0.000000001), -1,
                       1)  # TODO nekako boljše to rešit division by 0
        eopatch.add_feature(FeatureType.DATA, 'ARVI', arvi)
        arvi_slope = temporal_derivative(arvi.squeeze())
        # eopatch.add_feature(FeatureType.DATA, 'ARVI_SLOPE', arvi_slope[..., np.newaxis])

        evi = np.clip(2.5 * ((nir - red) / (nir + (self.c1 * red) - (self.c2 * blue) + self.L + 0.000000001)), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'EVI', evi)
        evi_slope = temporal_derivative(evi.squeeze())
        # eopatch.add_feature(FeatureType.DATA, 'EVI_SLOPE', evi_slope[..., np.newaxis])

        ndvi = np.clip((nir - red) / (nir + red + 0.000000001), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'NDVI', ndvi)
        ndvi_slope = temporal_derivative(ndvi.squeeze())
        # eopatch.add_feature(FeatureType.DATA, 'NDVI_SLOPE', ndvi_slope[..., np.newaxis])  # ASSUMES EVENLY SPACED

        band_a = eopatch.data['BANDS'][..., 1]
        band_b = eopatch.data['BANDS'][..., 3]
        ndvi = np.clip((band_a - band_b) / (band_a + band_b + 0.000000001), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'NDWI', ndvi[..., np.newaxis])

        sipi = np.clip((nir - blue) / (nir - red + 0.000000001), 0, 2)  # TODO nekako boljše to rešit division by 0
        eopatch.add_feature(FeatureType.DATA, 'SIPI', sipi)

        Lvar = 0.5
        savi = np.clip(((nir - red) / (nir + red + Lvar + 0.000000001)) * (1 + Lvar), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'SAVI', savi)

        img = np.clip(eopatch.data['BANDS'][..., [2, 1, 0]] * 3.5, 0, 1)
        t, w, h, _ = img.shape
        gray_img = np.zeros((t, w, h))
        for time in range(t):
            img0 = np.clip(eopatch[FeatureType.DATA]['BANDS'][time][..., [2, 1, 0]] * 3.5, 0, 1)
            img = rgb2gray(img0)
            gray_img[time] = (img * 255).astype(np.uint8)

        eopatch.add_feature(FeatureType.DATA, 'GRAY', gray_img[..., np.newaxis])

        return eopatch


class ValidDataFractionPredicate:
    """ Predicate that defines if a frame from EOPatch's time-series is valid or not. Frame is valid, if the
    valid data fraction is above the specified threshold.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return coverage > self.threshold


def apply_2d(lpis, func):
    width, height = lpis.shape
    found_nan = False
    for x in range(width):
        for y in range(height):
            if not found_nan:
                found_nan = True if np.isnan(lpis[x][y]) else False
            lpis[x][y] = func(lpis[x][y])
    return lpis, found_nan


class ModifyLPISTask(EOTask):

    def __init__(self, feature_name):
        self.feature_name = feature_name

    def execute(self, eopatch):
        all_features = eopatch.get_feature_list()
        if (FeatureType.MASK_TIMELESS, self.feature_name) in all_features:
            lpis = eopatch.mask_timeless[self.feature_name].squeeze()
            lpis, found = apply_2d(lpis, lambda x: 0 if np.isnan(x) else int(x + 1))
            if found:
                eopatch.add_feature(FeatureType.MASK_TIMELESS, self.feature_name, lpis[..., np.newaxis])
        else:
            t, w, h, _ = eopatch.data['BANDS'].shape
            eopatch.add_feature(FeatureType.MASK_TIMELESS, self.feature_name, np.zeros((w, h, 1)))
        return eopatch


if __name__ == '__main__':

    # no_patches = 1085
    no_patches = 1061

    # path = '/home/beno/Documents/test'
    path = 'E:/Data/PerceptiveSentinel'

    patch_location = path + '/Slovenia/'
    load = LoadTask(patch_location)

    save_path_location = path + '/Slovenia/'
    if not os.path.isdir(save_path_location):
        os.makedirs(save_path_location)

    save = SaveTask(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    # valid_data_predicate = ValidDataFractionPredicate(0.8)
    # filter_task = SimpleFilterTask((FeatureType.MASK, 'IS_VALID'), valid_data_predicate)
    # addStreamNDVI = AddStreamTemporalFeaturesTask(data_feature='NDVI')
    # addStreamSAVI = AddStreamTemporalFeaturesTask(data_feature='SAVI')
    # addStreamEVI = AddStreamTemporalFeaturesTask(data_feature='EVI')
    # addStreamARVI = AddStreamTemporalFeaturesTask(data_feature='ARVI')
    # addStreamSIPI = AddStreamTemporalFeaturesTask(data_feature='SIPI')
    # addStreamNDWI = AddStreamTemporalFeaturesTask(data_feature='NDWI')

    '''
    lulc_cmap = mpl.colors.ListedColormap([entry.color for entry in LULC])
    lulc_norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 11, 1), lulc_cmap.N)

    land_cover_path = path+'/shapefiles/slovenia.shp'

    land_cover = gpd.read_file(land_cover_path)

    land_cover_val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    land_cover_array = []
    for val in land_cover_val:
        temp = land_cover[land_cover.lulcid == val]
        temp.reset_index(drop=True, inplace=True)
        land_cover_array.append(temp)
        del temp

    rshape = (FeatureType.MASK, 'IS_VALID')

    land_cover_task_array = []
    for el, val in zip(land_cover_array, land_cover_val):
        land_cover_task_array.append(VectorToRaster(
            feature=(FeatureType.MASK_TIMELESS, 'LULC'),
            vector_data=el,
            raster_value=val,
            raster_shape=rshape,
            raster_dtype=np.uint8))
    '''
    execution_args = []
    for id in range(479, no_patches):
        execution_args.append({
            load: {'eopatch_folder': 'eopatch_{}'.format(id)},
            save: {'eopatch_folder': 'eopatch_{}'.format(id)}
        })
    # id = 2
    # execution_args = {
    #     load: {'eopatch_folder': 'eopatch_{}'.format(id)},
    #     save: {'eopatch_folder': 'eopatch_{}'.format(id)}
    # }

    addStreamGREEN = AddStreamTemporalFeaturesTask(data_feature='GREEN')
    addStreamBLUE = AddStreamTemporalFeaturesTask(data_feature='BLUE')

    cloud_classifier = get_s2_pixel_cloud_detector(all_bands=True)
    add_clm = AddCloudMaskTask(cloud_classifier, 'BANDS', cm_size_y=10, cm_size_x=10, cmask_feature='CLM',
                               cprobs_feature='CLP')

    linear_interp = LinearInterpolation(
        (FeatureType.DATA, 'BANDS'),  # name of field to interpolate
        mask_feature=(FeatureType.MASK, 'IS_VALID'),  # mask to be used in interpolation
        bounds_error=False  # extrapolate with NaN's
    )

    add_sh_valmask = AddValidDataMaskTask(SentinelHubValidData(),
                                          'IS_VALID'  # name of output mask
                                          )
    lpis_task = ModifyLPISTask('LPIS_2017')
    size_big = (500, 505)
    dem = SentinelHubDemTask((FeatureType.DATA_TIMELESS, 'DEM'), size=size_big)
    grad = AddGradientTask((FeatureType.DATA_TIMELESS, 'DEM'), (FeatureType.DATA_TIMELESS, 'INCLINATION'))

    workflow = LinearWorkflow(
        load,
        # printPatch(),
        # add_clm,
        # add_sh_valmask,
        # filter_task,
        # linear_interp,
        AddBaseFeatures(),
        dem,
        grad,
        # printPatch('base added '),
        # addStreamNDVI,
        # addStreamSAVI,
        # addStreamEVI,
        # addStreamARVI,
        # addStreamSIPI,
        # addStreamNDWI,
        # allValid('IS_VALID'),
        # *land_cover_task_array,
        # printPatch(),
        lpis_task,
        addStreamGREEN,
        addStreamBLUE,
        save
    )

    # workflow.execute(execution_args)

    # start_time = time.time()
    # runs workflow for each set of arguments in list
    executor = EOExecutor(workflow, execution_args, save_logs=True, logs_folder='ExecutionLogs')
    executor.run(workers=3, multiprocess=True)

    # file = open('stream_timing.txt', 'a')
    # running = str(dt.datetime.now()) + ' Running time: {}\n'.format(time.time() - start_time)
    # print(running)
    # file.write(running)
    # file.close()

    # executor.make_report()

    # workflow.execute({load: {'eopatch_folder': 'eopatch_1000'}, save: {'eopatch_folder': 'eopatch_1000'}, })
