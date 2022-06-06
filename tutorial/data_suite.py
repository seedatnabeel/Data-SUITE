import logging
import os
import sys

import numpy as np

from src.models.conformal import conformal_class
from src.models.copula import fit_sample_copula
from src.models.representation import representation_class_based
from src.utils.helpers import *

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Data_SUITE:
    def __init__(
        self,
        copula_type="vine",
        n_copula_samples=1000,
        representer="pca",
        rep_dim=None,
    ):
        """
        Args:
        copula_type (str): type of copula to use "vine" or "gaussian"
        n_copula_samples (int): number of copula samples
        representer (str): type of representer function 'pca' or 'ae'
        rep_dim (int): dimension of the representation (default=None)
        """

        self.copula_type = copula_type
        self.n_copula_samples = n_copula_samples
        self.representer = representer
        self.rep_dim = None

    def fit(self, train):
        """
        Fits Data SUITE to training data

        Args:
        train (numpy array): np array containing data
        """
        self.train = train
        self.suspect_features = list(range(train.shape[1]))

        self.copula()
        self.representer_fit()
        self.conformal_predictor_fit()

    def predict(self, test):
        """
        Predicts conformal intervals using Data SUITE on new test data.
        This can be processed for the users specific needs

        Args:
        test (numpy array): np array containing data
        """
        self.test = test
        self.representer_predict()
        self.conformal_predictor_predict()

        return self.conformal_dict, self.suspect_features

    def copula(self):
        """
        Fits and samples a copula
        """

        if self.copula_type != None:
            self.copula_samples = fit_sample_copula(
                clean_corpus=self.train,
                copula=self.copula_type,
                copula_n_samples=self.n_copula_samples,
            )
        else:
            self.copula_samples = self.train

    def representer_fit(self):
        """
        Fits a representer
        """

        if self.rep_dim == None:
            self.rep_dim = int(np.ceil(self.train.shape[1] / 2))

        (
            self.pcs_train,
            self.pcs_copula,
            self.representer,
            self.scaler,
        ) = representation_class_based(
            self.train,
            self.copula_samples,
            n_components=self.rep_dim,
            rep_type="pca",
        )

    def representer_predict(self):
        """
        Predicts with the representer
        """

        test_sc = self.scaler.transform(self.test)

        self.pcs_test = self.representer.transform(test_sc)

    def conformal_predictor_fit(self):
        """
        Fits a conformal predictor
        """

        self.conformal_predictors = []

        for feat in self.suspect_features:
            feat = int(feat)
            dim = self.pcs_copula.shape[1]
            conf = conformal_class(
                conformity_score="sign",
                input_dim=dim,
            )
            conf.fit(
                x_train=self.pcs_copula,
                y_train=self.copula_samples[:, feat],
            )

            self.conformal_predictors.append(conf)

    def conformal_predictor_predict(self):
        """
        Predicts using a conformal predictor
        """

        self.conformal_dict = {}

        for idx, feat in enumerate(self.suspect_features):
            feat = int(feat)
            dim = self.pcs_copula.shape[1]
            conf = self.conformal_predictors[idx]
            self.conformal_dict[feat] = conf.predict(
                x_test=self.pcs_test,
                y_test=self.test[:, feat],
            )
            logging.info(f"Running analysis for feature = {feat}")
