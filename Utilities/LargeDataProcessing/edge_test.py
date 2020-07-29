from edge_extraction import EdgeExtractionTask
import numpy as np
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk, EOPatch
import cv2
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = 'D:/Users/Beno/PycharmProjects/eo-learn/example_data/TestEOPatch'
    save_path = 'D:/Users/Beno/TestInputs/TestPatch4'
    eopatch = EOPatch.load(path)

    edges = EdgeExtractionTask({FeatureType.DATA: ['NDVI', 'BLUE']},
                               canny_low_threshold=0.15,
                               canny_high_threshold=0.30,
                               weight_threshold=0.05)

    eopatch.data['BLUE'] = eopatch.data['BANDS-S2-L1C'][..., 1, np.newaxis]

    eopatch = edges.execute(eopatch)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    eopatch.save(save_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    mask = eopatch.mask_timeless['EDGES_INV'].squeeze()
    print(mask[50])
    print(f'sum {np.sum(mask)}')

    plt.imshow(eopatch.mask_timeless['EDGES_INV'].squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.show()
