from edge_extraction import EdgeExtractionTask
import numpy as np
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk, EOPatch
import cv2
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = 'D:/Users/Beno/PycharmProjects/eo-learn/features/eolearn/tests/TestInputs'
    save_path = 'D:/Users/Beno/TestInputs'
    eopatch = EOPatch.load(f'{path}/TestPatch')

    edges = EdgeExtractionTask(['ndvi', 'random'])

    eopatch = edges.execute(eopatch)

    eopatch.save(f'{save_path}/TestPatch')

    plt.imshow(eopatch.mask_timeless, 'EDGES_INV')
    plt.show()
