#!usr/bin/env python3
"""
PCA based latent representation
"""

import numpy as np
from sklearn.decomposition import PCA

class PCA_Latent_Representation:
    """
    PCA based latent representation

    Args:
        n_components: Number of principal components to keep.
    """

    def __init__(self, n_components: int=2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model to the data.

        X: Data to fit the model to.
        """
        self.pca.fit(X)
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using the fitted PCA model.

        X: Data to transform
        return: Transformed data.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "PCA model has not been fitted yet. Call fit() before transform()."
                )

        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model to the data and transform it.

        X: Data to fit the model to and transform.
        return: Transformed data.
        """
        self.fit(X)
        self._is_fitted = True

        return self.transform(X)
    
    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """
        Get the explained variance ratio of the fitted PCA model.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "PCA model has not been fitted yet. Call fit() before accessing " \
                "explained_variance_ratio_."
                )

        return self.pca.explained_variance_ratio_
