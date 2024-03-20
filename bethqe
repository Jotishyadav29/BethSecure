import numpy as np
import os
import seaborn as sns
from sklearn.decomposition import PCA

BENCHMARK_LIST = ["rcov", "svm", "ifor", "dose"]

####################################################
## Custom Baseline Classes
####################################################

class WhitenedBenchmark():
    """
    Generic class to standardise scikit-learn model functions. Definitions for scikit-learn models available is in BENCHMARK.

    # TODO Approximation kernels: Nystroem, Random Fourier Features
    """
    def __init__(self, name, base_model, args):
        self.name = name # args.benchmark
        self.clf = base_model
        self.pca = PCA(whiten=True)

    def decision_function(self, X):
        return self.clf.decision_function(self.pca.transform(X))

    def fit(self, X):
        # Whiten the data using PCA
        self.pca = self.pca.fit(X)# .partial_fit(X)
        self.clf = self.clf.fit(self.pca.transform(X))

    def predict(self, X):
        return self.clf.predict(self.pca.transform(X))
