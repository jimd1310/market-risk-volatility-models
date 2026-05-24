"""
Helper functions for validating the HMM model.
"""
import numpy as np


def print_exp_durations(GHMM): 
    """
    Prints the expected durations for each hidden state in the HMM.

    Parameters
    ----------
    GHMM : instance
        The fitted GHMM model instance.
    """
    exp_durations = 1.0 / (1.0 - np.diag(GHMM.model.transmat_))
    print(f"{GHMM.n_states}-State HMM Expected Durations:")
    for i, duration in enumerate(exp_durations):
        print(f"State {i} Expected Duration: {duration:.6f}")


def print_params(GHMM):
    """
    Prints the parameters of the HMM model.

    Parameters
    ----------
    GHMM : instance
        The fitted GHMM model instance.
    """
    print(f"{GHMM.n_states}-State HMM Parameters:")
    print(f"{'State':<6}{'Mean':>10}{'Variance':>10}")
    for i in range(GHMM.n_states):
        mean = GHMM.model.means_[i][0]
        var = GHMM.model.covars_[i][0][0]
        print(f"{i:<6}{mean:>10.6f}{var:>10.6f}")


def print_state_stats(GHMM, logret): 
    """
    Prints the mean and std dev of observations based on the Viterbi-decoded
    state sequence.

    Parameters
    ----------
    GHMM : instance
        The fitted GHMM model instance.
    logret : NumPy array of shape (n_samples, 1)
        Log return time series data in percentage (x100).
    """
    viterbi_seq = GHMM.model.predict(logret)
    data = np.asarray(logret).ravel()

    print(f"{GHMM.n_states}-State HMM Sample Statistics:")
    print(f"{'State':<6}{'Count':>8}{'Mean':>12}{'Variance':>12}")
    
    counts = np.bincount(viterbi_seq)

    for state in range(GHMM.n_states):
        state_data = data[viterbi_seq == state]
        mean = np.mean(state_data)
        var = np.var(state_data, ddof=1)
        print(f"{state:<6}{counts[state]:>8}{mean:>12.6f}{var:>12.6f}")


def print_transition_matrix(GHMM): 
    """
    Prints the transition matrix of the HMM.

    Parameters
    ----------
    GHMM : instance
        The fitted GHMM model instance.
    """
    print(f"{GHMM.n_states}-State HMM Transition Matrix:")
    for i in range(GHMM.n_states):
        row = " ".join(f"{prob:.6f}" for prob in GHMM.model.transmat_[i])
        print(f"State {i}: {row}")