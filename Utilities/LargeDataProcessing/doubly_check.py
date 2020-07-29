import random
from eolearn.core import EOPatch, FeatureType

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from sentinelhub import geo_utils
from scipy.ndimage import gaussian_filter
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from eolearn.features.doubly_logistic_approximation import DoublyLogisticApproximationTask

if __name__ == '__main__':
    path = 'D:/Users/Beno/PycharmProjects/eo-learn/features/eolearn/tests/TestInputs'
    save_path = 'D:/Users/Beno/TestInputs'
    # patch_no = 578
    # eopatch1 = EOPatch.load(path + '/Slovenia/eopatch_{}'.format(patch_no), lazy_loading=True)
    # eopatch = EOPatch.load(path + '/LPIS/eopatch_{}'.format(patch_no), lazy_loading=True)
    # eopatch = EOPatch.load(f'{path}/eopatch_398')
    # lpis = eopatch.mask_timeless['LPIS_2017_G2'].squeeze
    eopatch = EOPatch.load(f'{path}/TestPatch')
    ndvi = eopatch.data['ndvi'].squeeze()
    mask = eopatch.mask['IS_VALID'].squeeze()
    for i, t in enumerate(eopatch.timestamp):
        plt.figure(str(t))
        plt.subplot(121)
        plt.imshow((ndvi[i]+1)*255/2, cmap='gray', vmin=0, vmax=255)
        plt.subplot(122)
        plt.imshow(mask[i]*255, cmap='gray', vmin=0, vmax=255)
        plt.show()
    doubly = DoublyLogisticApproximationTask(feature='ndvi', mask_feature=(FeatureType.MASK, 'IS_VALID'))
    eopatch = doubly.execute(eopatch)
    eopatch.save(f'{save_path}/TestPatch')
    # logi = eopatch.
    #
    # for x in range(20):
    #     for y in range(20):
    #         plt.subplot(211)
    #         plt.plot(ndvi[:, x, y])
    #         plt.subplot(212)
    #         plt.plot(mask[:, x, y])
    #         plt.show()
