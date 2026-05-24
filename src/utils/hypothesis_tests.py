import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.special import xlogy


def validate_breaches(breaches):
    b = np.asarray(breaches).ravel()

    if len(b) == 0:
        raise ValueError("breaches must be non-empty")

    if b.dtype == bool:
        return b.astype(int)

    if not np.all(np.isfinite(b)):
        raise ValueError("breaches contain NaN or inf")

    if not np.all(np.isin(b, [0, 1])):
        raise ValueError("breaches must contain only 0/1 or bool values")

    return b.astype(int)


def kupiec_test(breaches, VaR_conf=0.99, conf=0.95):
    hits = validate_breaches(breaches)

    T = len(hits)
    X = hits.sum()

    alpha = 1 - VaR_conf
    if not (0 < alpha < 1):
        raise ValueError("VaR_conf must be between 0 and 1")

    p_hat = X / T

    loglik_null = xlogy(X, alpha) + xlogy(T - X, 1 - alpha)
    loglik_alt = xlogy(X, p_hat) + xlogy(T - X, 1 - p_hat)

    LR = max(0.0, -2 * (loglik_null - loglik_alt))
    p_value = chi2.sf(LR, df=1)

    result = "Reject" if p_value < (1 - conf) else "Fail to reject"

    return {
        "n_breaches": int(X),
        "T": int(T),
        "breach_rate": X / T,
        "expected_rate": alpha,
        "LR_uc": LR,
        "p_uc": p_value,
        "uc_result": result
    }


def christoffersen_independence_test(breaches, conf=0.95):
    hits = validate_breaches(breaches)

    if len(hits) < 2:
        raise ValueError("Need at least 2 observations")

    prev = hits[:-1]
    curr = hits[1:]

    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))

    den0 = n00 + n01
    den1 = n10 + n11

    p01 = n01 / den0 if den0 > 0 else 0.0
    p11 = n11 / den1 if den1 > 0 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    loglik_null = (
        xlogy(n01 + n11, p)
        + xlogy(n00 + n10, 1 - p)
    )

    loglik_alt = (
        xlogy(n01, p01)
        + xlogy(n00, 1 - p01)
        + xlogy(n11, p11)
        + xlogy(n10, 1 - p11)
    )

    LR = max(0.0, -2 * (loglik_null - loglik_alt))
    p_value = chi2.sf(LR, df=1)

    result = "Reject" if p_value < (1 - conf) else "Fail to reject"

    return {
        "n00": int(n00),
        "n01": int(n01),
        "n10": int(n10),
        "n11": int(n11),
        "p01": p01,
        "p11": p11,
        "LR_ind": LR,
        "p_ind": p_value,
        "ind_result": result
    }