"""
Module for extraction of edge mask in EOPatch
"""
import numpy as np
from eolearn.core import EOTask, FeatureType
import cv2


class EdgeExtractionTask(EOTask):
    """
    Task computes a timeless mask of of edges from single patch based on multiple features during whole year.

    Mask is computed in several steps:
        - Individual image edge calculation
            Each image is firstly blurred with a Gaussian filter (cv2.GaussianBlur), then edges are computed using
            edge detector (cv2.Canny), finally dilation and erosion are applied for filling potential holes.
        - Individual image weight calculation
            Each edge pixel's contribution is adjusted based on that feature's values in the vicinity. The weights are
            calculated by normalizing and blurring image with a Gaussian filter (cv2.GaussianBlur).
        - Yearly feature mask calculation by joining single weighted images for each feature
            Weight mask is calculated by summing all weights for each pixel through the whole year.
        - Final temporal mask calculation by joining all the yearly feature masks
            Pixels are included only, if total sum of all feature's weights for that pixel exceeds the threshold.
    """

    def __init__(self,
                 features,
                 output_feature='EDGES_INV',
                 canny_low_threshold=40,
                 canny_high_threshold=80,
                 canny_blur_size=5,
                 canny_blur_sigma=2,
                 structuring_element=None,
                 dilation_mask_size=3,
                 erosion_mask_size=2,
                 weight_blur_size=7,
                 weight_blur_sigma=4,
                 weight_threshold=0.05,
                 mask_feature=None):
        """
        :param features: A list of feature names from which the edges are computed. All the features need to be of type
            FeatureType.Data
        :type features: list of str
        :param output_feature: Name of output feature
        :param output_feature: str
        :param canny_low_threshold:
        :param canny_high_threshold:
        :param canny_blur_size:
        :param canny_blur_sigma:
        :param structuring_element:
        :param dilation_mask_size:
        :param erosion_mask_size:
        :param weight_blur_size:
        :param weight_blur_sigma:
        :param weight_threshold:
        :param mask_feature:
        """

        self.features = features
        self.output_feature = output_feature
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.canny_blur = lambda x: cv2.GaussianBlur(x, (canny_blur_size, canny_blur_size), canny_blur_sigma)
        if not structuring_element:
            self.structuring_element = [[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]]
        else:
            self.structuring_element = structuring_element
        self.dilation_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_mask_size, dilation_mask_size))
        self.erosion_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_mask_size, erosion_mask_size))
        self.weight_blur = lambda x: cv2.GaussianBlur(x, (weight_blur_size, weight_blur_size), weight_blur_sigma)
        self.weight_threshold = weight_threshold
        self.mask_feature = mask_feature

    @staticmethod
    def normalization(feature):
        f_min = np.min(feature)
        f_max = np.max(feature)
        return (feature - f_min) / (f_max - f_min)

    def execute(self, eopatch):

        first_feature = eopatch.data[self.features[0]]
        timestamps, width, height, _ = first_feature.shape

        no_feat = len(self.features)
        all_edges = np.zeros((no_feat, timestamps, width, height))

        for i, feature in enumerate(self.features):
            images = eopatch.data[feature].squeeze()
            images = self.normalization(images)
            feature_edge = np.zeros((timestamps, width, height))
            for individual_time in range(timestamps):
                one_image = images[individual_time]
                smoothed_image = self.canny_blur(one_image)
                one_edge = cv2.Canny(smoothed_image.astype(np.uint8),
                                     self.canny_low_threshold,
                                     self.canny_high_threshold)
                feature_edge[individual_time] = one_edge > 0

            adjust_weights = [self.weight_blur(x) for x in images]
            all_edges[i] = feature_edge * adjust_weights

        edge_vector = np.sum(all_edges, (0, 1))
        edge_vector = edge_vector / (timestamps * len(self.features))
        all_edges = edge_vector > self.weight_threshold

        all_edges = 1 - all_edges
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.output_feature, all_edges[..., np.newaxis])
        return eopatch
