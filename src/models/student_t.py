"""
i.i.d. Student-t model for benchmarking purposes.

This module implements an i.i.d. Student-t model on 100 x log(returns).
"""
from .base import Base
import numpy as np
from scipy.stats import t


class StudentT(Base): 
    """
    Creates an instance of i.i.d. Student-t model for volatility benchmarking.
    """
    def __init__(self):
        self.df = None
        self.mu = None
        self.sigma = None
        self.loglik = None
        self.n_params = 3  # df, mu, and sigma


    def fit(self, logret):
        """
        Fits the i.i.d. Student-t model to log return data.

        Parameters
        ----------
        logret : NumPy array of shape (n_samples, 1)
            Log return time series data in percentage (x100).
        
        Returns
        -------
        self : StudentT
            The fitted StudentT instance.
        """
        data = np.asarray(logret).ravel()

        self.df, self.mu, self.sigma = t.fit(data)
        self.loglik = np.sum(
            t.logpdf(data, df=self.df, loc=self.mu, scale=self.sigma)
        )

        return self


    def log_predictive_density(self, logret_test):
        """
        Computes the log predictive density of the test data.

        Parameters
        ----------
        logret_test : NumPy array of shape (n_samples, 1)
            Test log return time series data in percentage (x100).

        Returns
        -------
        log_scores : NumPy array of shape (n_samples, 1)
            Log predictive densities for each test data point.
        """
        data = np.asarray(logret_test).ravel()
        log_scores = t.logpdf(data, df=self.df, loc=self.mu, scale=self.sigma)
        return np.array(log_scores)
        

    def VaR_forecast(self, confidence, logret_test): 
        return np.full(
            len(logret_test),
            -(self.mu + self.sigma * t.ppf(1 - confidence, df=self.df))
        )


    def exp_shortfall_forecast(self, confidence, logret_test):
        q = t.ppf(1 - confidence, df=self.df)
        return np.full(
            len(logret_test),
            (-self.mu 
              + self.sigma * (self.df + q**2) / (self. df - 1)
              * t.pdf(q, df=self.df) / (1 - confidence))
        )