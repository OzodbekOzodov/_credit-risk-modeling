# This file contains user-defined functions and docstrings used throughout the project

import pandas as pd
import numpy as np
from sklearn import linear_model
import scipy.stats as stat



class LogisticRegression_p_values:

    def _init_(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)

    def fit(self, x, y):
        self.model.fit(x,y)
        denominator = (2.0*(2.0 + np.cosh(self.model.decision_function(x))))
        denominator = np.title(denominator, (x.shape[1],1)).T
        F_ij = np.dot((x/denominator).T,x)
        Cramer_Rao = np.linalg.inv(F_ij)
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values 