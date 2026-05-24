"""
GARCH(1,1) model for time-varying volatility, used for benchmarking.

This module implements a GARCH(1,1) model on 100 x log(returns).
"""
from .base import Base
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from arch import arch_model 


class GARCH(Base):
    """
    Creates an instance of GARCH(1,1) model for volatility benchmarking.
    Can specify either Student-t or Gaussian innovations.
    """
    def __init__(self, p=1, q=1, distribution='normal'):
        self.model = None
        self.res = None
        self.loglik = None
        self.n_params = None
        self.train_data = None
        self.p = p
        self.q = q
        self.distribution = distribution


    def fit(self, logret):
        """
        Fits the GARCH(1,1) model to log return data.

        Parameters
        ----------
        logret : NumPy array of shape (n_samples, 1)
            Log return time series data in percentage (x100).
        
        Returns
        -------
        self : GARCH
            The fitted GARCH instance.
        """
        data = np.asarray(logret, dtype=float).ravel()
        self.train_data = data

        self.model = arch_model(
            data, 
            mean='zero', 
            vol='GARCH', 
            p=self.p, 
            q=self.q, 
            dist=self.distribution
        )
        
        self.res = self.model.fit(disp="off")
        self.loglik = self.res.loglikelihood
        self.n_params = len(self.res.params)

        return self
    

    def volatility_forecast(self, logret_test): 
        """
        Computes one-step-ahead conditional volatility forecasts over test data.

        Parameters
        ----------
        logret_test : NumPy array of shape (n_samples, 1)
            Test log return time series data in percentages.
        
        Returns 
        -------
        sigmas : NumPy array of shape (n_samples, )
            One-step-ahead conditional volatility forecasts.
        """
        data = np.asarray(logret_test, dtype=float).ravel()

        params = self.res.params 
        omega = params['omega']

        # Most recent returns and conditional variance 
        y_lags = list(self.train_data[-self.p:][::-1])

        sigma2_train = np.asarray(self.res.conditional_volatility) ** 2
        sigma2_lags = list(sigma2_train[-self.q:][::-1])

        sigmas = []

        for y in data: 
            sigma2 = omega

            for i in range(1, self.p + 1): 
                sigma2 += params[f'alpha[{i}]'] * y_lags[i - 1] ** 2

            for j in range(1, self.q + 1): 
                sigma2 += params[f'beta[{j}]'] * sigma2_lags[j - 1]
            
            sigma = np.sqrt(sigma2)
            sigmas.append(sigma)

            y_lags = [y] + y_lags[:-1]
            sigma2_lags = [sigma2] + sigma2_lags[:-1]

        return np.array(sigmas)
    

    def student_t_scale(self):
        """
        Returns the scale adjustment for the standardised student-t distribution.

        If X ~ t_nu, then sqrt((nu - 2) / nu) X has variance 1. 
        """
        nu = self.res.params['nu']
        return np.sqrt((nu - 2) / nu)
    
    
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
        sigmas = self.volatility_forecast(logret_test)

        log_scores = []

        if self.distribution == 'normal': 
            for y, sigma in zip(data, sigmas): 
                log_scores.append(norm.logpdf(y, loc=0, scale=sigma))
        else: 
            nu = self.res.params['nu']
            scale = self.student_t_scale()

            for y, sigma in zip(data, sigmas): 
                log_scores.append(
                    t.logpdf(y, df=nu, loc=0, scale=sigma * scale)
                )

        return np.array(log_scores)
    

    def VaR_forecast(self, confidence, logret_test): 
        """
        Computes one-step-ahead VaR forecasts under the fitted GARCH model.

        Parameters
        ----------
        confidence : float
            VaR confidence level. For example, 0.99 for 99% VaR.

        logret_test : NumPy array of shape (n_samples, 1)
            Test log return time series data in percentage (x100).

        Returns
        -------
        VaR : NumPy array of shape (n_samples,)
            One-step-ahead VaR forecasts for losses, where loss = -return.
        """
        sigmas = self.volatility_forecast(logret_test)
        alpha = 1 - confidence

        if self.distribution == 'normal': 
            q = norm.ppf(alpha)
            return -sigmas * q
        
        else: 
            nu = self.res.params['nu']
            scale = self.student_t_scale()
            q = t.ppf(alpha, df=nu)

            return -sigmas * scale * q


    def exp_shortfall_forecast(self, confidence, logret_test):
        """
        Computes one-step-ahead Expected Shortfall forecasts under the fitted
        GARCH model.

        Parameters
        ----------
        confidence : float
            ES confidence level. For example, 0.99 for 99% ES.

        logret_test : NumPy array of shape (n_samples, 1)
            Test log return time series data in percentage (x100).

        Returns
        -------
        ES : NumPy array of shape (n_samples,)
            One-step-ahead Expected Shortfall forecasts for losses,
            where loss = -return.
        """
        sigmas = self.volatility_forecast(logret_test)
        alpha = 1 - confidence 

        if self.distribution == 'normal': 
            q = norm.ppf(alpha)
            return sigmas * norm.pdf(q) / alpha
        
        else: 
            nu = self.res.params['nu']
            scale = self.student_t_scale()

            q = t.ppf(alpha, df=nu)
            pdf_q = t.pdf(q, df=nu)

            es_multiplier = (
                scale
                * ((nu + q ** 2) / (nu - 1))
                * (pdf_q / alpha)
            )
            
            return sigmas * es_multiplier