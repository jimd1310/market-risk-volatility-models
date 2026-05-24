"""
Hidden Markov Model for modelling Volatility Regimes.

This module implements a Gaussian HMM on 100 x log(returns).
"""
from .base import Base
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.optimize import root_scalar
from hmmlearn import hmm


class GHMM(Base):
    """
    Creates an instance of the Gaussian HMM for volatility regime detection.

    Parameters
    ---------------------
    n_states : int, default=2
        Number of hidden states in the HMM.
    
    random_state : int or None, default=None
        Random seed for reproducibility.
    """
    def __init__(self, n_states=2, random_state=None):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.loglik = None
        self.train_data = None
        self.n_params = (
            self.n_states - 1 + # initial state distribution
            self.n_states * (self.n_states - 1) + # transition matrix
            self.n_states * 2 # means and variances
        )
    

    def fit(self, logret):
        """
        Fits the HMM to log return data.

        Parameters
        ----------
        logret : NumPy array of shape (n_samples, 1)
            Log return time series data in percentage (x100).

        Returns
        -------
        self : GHMM
            The fitted GHMM instance.
        """
        data = np.asarray(logret).reshape(-1, 1)
        self.train_data = data

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            n_iter=1000,
            covariance_type='diag',
            tol=1e-4,
            random_state=self.random_state
        )
        self.model.fit(data)
        self.loglik = self.model.score(data)
        
        return self
    

    def params(self):
        means = self.model.means_.flatten()
        stds = np.sqrt(self.model.covars_.flatten())
        return means, stds


    def cdf(self, x, weights): 
        """
        The unconditional cdf of the HMM, a mixture of Gaussian cdf's. 

        Parameters
        ---------- 
        x : float
            A given quantile. 
        
        Returns 
        -------
        The cdf evaluated at x. 
        """
        means, stds = self.params()
        weights = np.asarray(weights)
        weights = weights / weights.sum()

        return np.sum(weights * norm.cdf(x, loc=means, scale=stds))
    

    def log_predictive_density(self, logret_test): 
        """
        Compute the log predictive density of the test data.

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

        P = self.model.transmat_
        means, stds = self.params()

        # Start from training probabilities
        gamma_train = self.model.predict_proba(
            np.asarray(self.train_data).reshape(-1, 1)
            )
        gamma = gamma_train[-1]

        log_scores = []

        for y in data: 
            pi_pred = gamma @ P

            log_components = (
                np.log(np.clip(pi_pred, 1e-300, 1.0))
                + norm.logpdf(y, loc=means, scale=stds)
            )

            log_f = logsumexp(log_components)
            log_scores.append(log_f)

            gamma = np.exp(log_components - log_f)

        return np.array(log_scores)
    

    def VaR_given_weights(self, confidence, weights): 
        """
        One-step-ahead predictive VaR. 
        """
        alpha = 1 - confidence 
        means, stds = self.params() 

        lower = np.min(means - 20 * stds)
        upper = np.max(means + 20 * stds)

        def objective(q): 
            return self.cdf(q, weights) - alpha
        
        sol = root_scalar(objective, bracket=[lower, upper], method="brentq")

        if not sol.converged: 
            raise RuntimeError("VaR root finding failed to converge.")
        
        q_alpha = sol.root 

        return -q_alpha
    
    
    def VaR(self, confidence): 
        P = self.model.transmat_ 

        gamma_train = self.model.predict_proba(
            np.asarray(self.train_data).reshape(-1, 1)
        )

        gamma_T = gamma_train[-1]
        weights = gamma_T @ P

        return self.VaR_given_weights(confidence, weights)
    

    def VaR_forecast(self, confidence, logret_test): 
        """
        Dynamic one-step-ahead VaR forecasts over the test period
        """
        data = np.array(logret_test).ravel()

        P = self.model.transmat_ 
        means, stds = self.params() 

        gamma_train = self.model.predict_proba(
            np.asarray(self.train_data).reshape(-1, 1)
        )
        gamma = gamma_train[-1]

        VaRs = []

        for y in data: 
            # Forecast before observing y 
            weights = gamma @ P 
            VaR_t = self.VaR_given_weights(confidence, weights)
            VaRs.append(VaR_t)

            # Update 
            log_components = (
                np.log(np.clip(weights, 1e-300, 1.0))
                + norm.logpdf(y, loc=means, scale=stds)
            )

            log_f = logsumexp(log_components)
            gamma = np.exp(log_components - log_f)

        return np.array(VaRs)
    

    def exp_shortfall_given_weights(self, confidence, weights):

        weights = np.asarray(weights)
        weights = weights / weights.sum()

        alpha = 1 - confidence 
        means, stds = self.params() 

        q_alpha = -self.VaR_given_weights(confidence, weights)

        z = (q_alpha - means) / stds

        # E[r 1{r <= q_alpha}] for each Gaussian state 
        tail_first_moment = means * norm.cdf(z) - stds * norm.pdf(z)

        # E[r | r <= q_alpha] under the mixture 
        tail_mean_return = np.sum(weights * tail_first_moment) / alpha 

        return - tail_mean_return 
    

    def exp_shortfall(self, confidence):
        P = self.model.transmat_ 

        gamma_train = self.model.predict_proba(
            np.asarray(self.train_data).reshape(-1, 1)
        )

        gamma_T = gamma_train[-1]
        weights = gamma_T @ P 

        return self.exp_shortfall_given_weights(confidence, weights)
    

    def exp_shortfall_forecast(self, confidence, logret_test): 
        data = np.asarray(logret_test).ravel() 

        P = self.model.transmat_ 
        means, stds = self.params() 

        gamma_train = self.model.predict_proba(
            np.asarray(self.train_data).reshape(-1, 1)
        )
        gamma = gamma_train[-1]

        ES = []

        for y in data: 
            # Forecast 
            weights = gamma @ P 
            ES_t = self.exp_shortfall_given_weights(confidence, weights)
            ES.append(ES_t)

            # update
            log_components = (
                np.log(np.clip(weights, 1e-300, 1.0))
                + norm.logpdf(y, loc=means, scale=stds)
            )

            log_f = logsumexp(log_components)
            gamma = np.exp(log_components - log_f)

        return np.array(ES)
    
    def volatility_forecast(self, logret_test):
        data = np.asarray(logret_test).ravel()

        P = self.model.transmat_
        means, stds = self.params()

        gamma_train = self.model.predict_proba(
            np.asarray(self.train_data).reshape(-1, 1)
        )
        gamma = gamma_train[-1]

        vols = []

        for y in data:
            weights = gamma @ P

            mix_mean = np.sum(weights * means)
            mix_second_moment = np.sum(weights * (stds**2 + means**2))
            mix_var = mix_second_moment - mix_mean**2

            vols.append(np.sqrt(mix_var))

            log_components = (
                np.log(np.clip(weights, 1e-300, 1.0))
                + norm.logpdf(y, loc=means, scale=stds)
            )
            log_f = logsumexp(log_components)
            gamma = np.exp(log_components - log_f)

        return np.array(vols)

                               
