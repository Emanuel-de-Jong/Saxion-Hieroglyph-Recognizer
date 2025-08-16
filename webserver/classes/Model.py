import os

import joblib
import numpy as np

from .FeatureExtractor import FeatureExtractor
from .Preprocessor import Preprocessor

class Model:
    labels = ['Ing', 'Tyr', 'earth', 'elk-sedge', 'game',
              'hail', 'ice', 'joy', 'lake', 'wealth', 'year']

    def __init__(self):
        self.model = joblib.load(f'models{os.sep}best_model.joblib')

        self.feature_extractor = FeatureExtractor()
        self.preprocessor = Preprocessor()
    
    def predict(self, img_path):
        self.features = self.feature_extractor.extract(img_path)
        if self.features is None:
            return None
        
        self.features = self.preprocessor.process(self.features)

        self.features = self.features[self.model.feature_names_in_]

        prediction = self.model.predict(self.features)
        label = self.labels[prediction[0]]
        
        return label
    
    def get_accuracy(self):
        accuracy = 1.0
        if hasattr(self.model, "predict_proba"):
            print("predict_proba")
            probabilities = self.model.predict_proba(self.features)
            accuracy = np.max(probabilities)
        elif hasattr(self.model, "decision_function"):
            print("decision_function")
            decision_scores = self.model.decision_function(self.features)
            accuracy = np.max(decision_scores)
        
        return accuracy
