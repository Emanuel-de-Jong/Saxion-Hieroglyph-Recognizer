import os
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Preprocessor:
    def __init__(self):
        with open(f'models{os.sep}min_max_scaler.pkl', 'rb') as f:
            self.min_max_scaler = pickle.load(f)
        
        with open(f'models{os.sep}z_score_scaler.pkl', 'rb') as f:
            self.z_score_scaler = pickle.load(f)

    def process(self, features):
        features_to_min_max = ['height', 'width', 'centroid_x', 'centroid_y', 'extent', 'perimeter']
        features = self.scale(features, features_to_min_max, self.min_max_scaler)

        features_to_z_score = ['area', 'hole_count', 'convex_area', 'solidity', 'eccentricity', 'orientation',
                               'hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7', 'top_bottom_ratio', 'height_width_ratio']
        features = self.scale(features, features_to_z_score, self.z_score_scaler)
        
        return features
    
    def scale(self, features, features_to_scale, scaler):
        to_scale = features[features_to_scale]
        scaled_values = scaler.transform(to_scale)
        features[features_to_scale] = scaled_values

        return features
