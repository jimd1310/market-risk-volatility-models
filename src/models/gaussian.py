"""
i.i.d. Gaussian model for benchmarking purposes.

This module implements an i.i.d. Gaussian (Normal) model on 100 x log(returns).
"""
from .base import Base
import numpy as np
from scipy.stats import norm


class Gaussian(Base): 
    """
    Creates an instance of i.i.d. Gaussian model for volatility benchmarking.
    """
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.loglik = None
        self.n_params = 2  # mu and sigma


    def fit(self, logret):
        """
        Fits the i.i.d. Gaussian model to log return data.

        Parameters
        ----------
        logret : NumPy array of shape (n_samples, 1)
            Log return time series data in percentage (x100).
        
        Returns
        -------
        self : Gaussian
            The fitted Gaussian instance.
        """
        data = np.asarray(logret).ravel()

        self.mu, self.sigma = norm.fit(data)
        self.loglik = np.sum(norm.logpdf(data, loc=self.mu, scale=self.sigma))

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
        log_scores = norm.logpdf(data, loc=self.mu, scale=self.sigma)
        
        return np.array(log_scores)
    
    def VaR_forecast(self, confidence, logret_test): 
        return np.full(
            len(logret_test), 
            -norm.ppf(1 - confidence, loc = self.mu, scale = self.sigma)
        )
    
    def exp_shortfall_forecast(self, confidence, logret_test): 
        return np.full(
            len(logret_test), 
            (-self.mu 
            + self.sigma * norm.pdf(norm.ppf(1 - confidence)) / (1 - confidence))
        )
