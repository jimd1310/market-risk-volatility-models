"""
The base class for all models. Provides common functionality and interfaces.
"""
from abc import ABC, abstractmethod


class Base(ABC): 
    """
    Abstract interface for all models.
    """

    @abstractmethod
    def fit(self, data_train): 
        """
        Fit the model to training data.

        Parameters
        ----------
        data_train : array-like
            The training data to fit the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    
    @abstractmethod
    def log_predictive_density(self, data_test): 
        """
        Compute the log predictive density of the test data.

        Parameters
        ----------
        data_test : array-like
            The test data for which to compute the log predictive density.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def VaR_forecast(self, confidence, data_test): 
        """
        Compute the value-at-risk (VaR) for a given confidence level.

        Parameters
        ----------
        confidence : float
            Confidence level, e.g. 99% confidence corresponds to 99% VaR.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def exp_shortfall_forecast(self, confidence, data_test): 
        """
        Compute the expected shortfall (ES) at a given confidence level.

        Parameters
        ----------
        confidence : float 
            Confidence level, e.g. 99% confidence corresponds to 99% ES.
        """
        raise NotImplementedError("Subclasses must implement this method.")

