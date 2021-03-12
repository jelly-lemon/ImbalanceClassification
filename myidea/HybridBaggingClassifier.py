from myidea.AdaSamplingBaggingClassifier import AdaSamplingBaggingClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class hybridBaggingClassifier():
    def __init__(self, n_undersampling_classifier, n_upsampling_classifier):
        self.under_classifier = AdaSamplingBaggingClassifier(n_undersampling_classifier)
        self.up_classifier = AdaSamplingBaggingClassifier(n_upsampling_classifier)

    def fit(self, x, y):
        self.under_classifier.fit(x, y, "under", show_info=True)
        self.up_classifier.fit(x, y, "up", show_info=True)

    def predict_proba_2(self, x):
        all_proba_1 = self.under_classifier.predict_proba_2(x)
        all_proba_2 = self.up_classifier.predict_proba_2(x)
        all_proba = np.concatenate((all_proba_1, all_proba_2), axis=0)

        return all_proba