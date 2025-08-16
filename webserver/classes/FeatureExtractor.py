from collections import defaultdict

import numpy as np
import pandas as pd
from skimage import io, morphology, measure

class FeatureExtractor:
    def extract(self, img_path):
        img = io.imread(img_path, as_gray=True)

        img = img > 0.99
        img = morphology.binary_erosion(img, morphology.disk(1))

        img_label = measure.label(img, background=1, connectivity=2)
        img_label = (img_label > 0).astype(int)

        regions = measure.regionprops(img_label)
        if len(regions) == 0:
            return None

        region = regions[0]

        feature_dict = defaultdict(float)
        feature_dict['area'] = region.area
        feature_dict['width'] = region.bbox[3] - region.bbox[1]
        feature_dict['height'] = region.bbox[2] - region.bbox[0]
        feature_dict['hole_count'] = 1 - region.euler_number
        feature_dict['centroid_x'] = region.centroid[0]
        feature_dict['centroid_y'] = region.centroid[1]
        feature_dict['extent'] = region.extent
        feature_dict['convex_area'] = region.convex_area
        feature_dict['solidity'] = region.solidity
        feature_dict['eccentricity'] = region.eccentricity
        feature_dict['orientation'] = region.orientation
        feature_dict['perimeter'] = region.perimeter
        feature_dict['hu_1'] = region.moments_hu[0]
        feature_dict['hu_2'] = region.moments_hu[1]
        feature_dict['hu_3'] = region.moments_hu[2]
        feature_dict['hu_4'] = region.moments_hu[3]
        feature_dict['hu_5'] = region.moments_hu[4]
        feature_dict['hu_6'] = region.moments_hu[5]
        feature_dict['hu_7'] = region.moments_hu[6]

        half_height = img.shape[0] // 2
        top = img[:half_height]
        bottom = img[half_height:]
        top_pixel_count = np.sum(top == False)
        bottom_pixel_count = np.sum(bottom == False)
        feature_dict['top_bottom_ratio'] = top_pixel_count - bottom_pixel_count

        feature_dict['height_width_ratio'] = (region.bbox[2] - region.bbox[0]) / (region.bbox[3] - region.bbox[1])

        return pd.DataFrame([feature_dict])
