"""
A function for fitting the HMM across multiple seeds.
"""
import numpy as np


def fit_best(GHMM, data, n_seeds, **model_kwargs):
    """
    Fits a model across multiple random seeds and selects the best fit
    based on the highest log-likelihood.

    Parameters
    ----------
    GHMM : class
        The GHMM model class to be instantiated and fitted.
    data : array-like
        The data to fit the model to.
    n_seeds : int
        The number of different random seeds to try.
    **model_kwargs : dict
        Additional keyword arguments to pass to the model constructor.

    Returns
    -------
    best_hmm : instance
        The best fitted model instance.
    """
    best_loglik = -np.inf
    best_hmm = None

    for seed in range(n_seeds):
        hmm = GHMM(random_state=seed, **model_kwargs)
        hmm.fit(data)

        if hmm.loglik > best_loglik:
            best_loglik = hmm.loglik
            best_hmm = hmm

    return best_hmm