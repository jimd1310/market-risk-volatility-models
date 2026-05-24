"""
Plots for the HMM. 
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def plot_densities(GHMM, logret): 
    """
    Plots the conditional density functions for each hidden state in the HMM.

    Parameters
    ----------
    GHMM : instance
        The fitted GHMM model instance.
    logret : NumPy array of shape (n_samples, 1)
        Log return time series data in percentage (x100).
    """
    y = logret.values
    T = y.size 

    means = GHMM.model.means_.flatten()
    stds = np.sqrt(GHMM.model.covars_).flatten()
    viterbi_seq = GHMM.model.predict(y.reshape(-1, 1))

    for k in range(GHMM.n_states): 
        mask = viterbi_seq == k
        plt.figure(figsize=(9, 3))
        plt.title(f'State {k} Density Plot')
        plt.hist(y[mask], 
                 bins=max(15, int(np.sqrt(mask.sum()))), 
                 density=True, 
                 alpha=0.5, 
                 color='gray', 
                 label='Empirical Density'
        )
        x = np.linspace(y.min(), y.max(), 1000)
        plt.plot(x, 
                 norm.pdf(x, loc=means[k], scale=stds[k]), 
                 color=f'C{k}', 
                 lw=3, 
                 label='Conditional Density'
        )
        plt.xlabel('Log Returns (%)')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.show()


def plot_viterbi_seq(GHMM, logret): 
    """
    Plots the Viterbi sequence of hidden states over time.

    Parameters
    ----------
    GHMM : instance
        The fitted GHMM model instance.
    logret : NumPy array of shape (n_samples, 1)
        Log return time series data in percentage (x100).
    """
    y = logret.values
    viterbi_seq = GHMM.model.predict(y.reshape(-1, 1))

    plt.figure(figsize=(9, 3))
    plt.plot(logret.index, 
             y, color='black', 
             alpha=0.3, 
             lw=0.8, 
             label='Log Returns'
    )
    for i in range(GHMM.n_states): 
        mask = viterbi_seq == i
        plt.scatter(logret.index[mask], 
                    y[mask], 
                    s=8, 
                    label=f'State {i}'
        )
    plt.title('Viterbi-Decoded Sequence of Hidden States')
    plt.xlabel('Time')
    plt.ylabel('Log Returns (%)')
    plt.legend(loc='upper right')
    plt.show()


def plot_probs(GHMM3, logret): 
    """
    Plots the smoothed probabilities of each hidden state over time.

    Parameters
    ----------
    GHMM : instance
        The fitted GHMM model instance.
    logret : NumPy array of shape (n_samples, 1)
        Log return time series data in percentage (x100).
    """
    y = logret.values
    smoothed_probs = GHMM3.model.predict_proba(y.reshape(-1, 1))

    plt.figure(figsize=(9, 3))
    for i in range(GHMM3.n_states): 
        plt.plot(logret.index, 
                 smoothed_probs[:, i], 
                 lw=0.8, 
                 label=f'State {i}'
        )
    plt.title('Smoothed Probabilities of Hidden States')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    plt.show()
