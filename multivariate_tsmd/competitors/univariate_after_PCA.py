import numpy as np
from sklearn.decomposition import PCA

class UnivariateAfterPCA(object):

    def __init__(self, univariate_method, method_params) -> None:
        self.method=univariate_method(**method_params)

    def fit(self,signal:np.ndarray)->None:
        pca=PCA(n_components=1)
        self.transformed_signal=np.hstack(pca.fit_transform(signal))
        self.method.fit(self.transformed_signal)
    
    @property
    def prediction_mask_(self):
        return self.method.prediction_mask_