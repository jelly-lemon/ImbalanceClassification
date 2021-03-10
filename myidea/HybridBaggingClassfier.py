from myidea.AdaSamplingBaggingClassifier import AdaSamplingBaggingClassifier


class hybridBaggingClassifier():
    def __init__(self, n_undersampling_classifier, n_upsampling_classifier):
        
        under_classifier = AdaSamplingBaggingClassifier(n_undersampling_classifier)
        up_classifier = AdaSamplingBaggingClassifier(n_upsampling_classifier)
