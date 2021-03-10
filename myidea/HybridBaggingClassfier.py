from myidea.AdaSamplingBaggingClassifier import AdaSamplingBaggingClassifier


class hybridBaggingClassifier():
    def __init__(self, n_undersampling_classifier):
        
        undersampling_classifier = AdaSamplingBaggingClassifier(n_undersampling_classifier)