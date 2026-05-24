"""
Information criteria for model selection.
"""
import numpy as np


def aic(loglik, n_params):
    """
    Calculates Akaike Information Criterion.

    Parameters
    ----------
    loglik : float
        Log-likelihood of the model.
    n_params : int 
        Number of parameters in the model.

    Returns
    -------
    Float : 
        AIC value.
    """
    return 2 * n_params - 2 * loglik


def bic(loglik, n_params, n_samples):
    """
    Calculates Bayesian Information Criterion.

    Parameters
    ----------
    loglik : float
        Log-likelihood of the model.
    n_params : int 
        Number of parameters in the model.
    n_samples : int
        Number of data samples.

    Returns
    -------
    float : 
        BIC value.
    """
    return n_params * np.log(n_samples) - 2 * loglik