import numpy as np 
import matplotlib.pyplot as plt

def plot_breaches(
    VaR_results, 
    breach_results, 
    logret_test, 
    model_names=None, 
    level_idx=1, 
    confidences=(0.95, 0.99), 
    figsize=(12, 5)
):
    """
    Plot return series, VaR threshold, and VaR breaches for each model.

    Parameters
    ----------
    VaR_results : dict
        model_name -> array of shape (2, T), where row 0 is 95% VaR and row 1 is 99% VaR.

    breach_results : dict
        model_name -> boolean array of shape (2, T).

    logret_test : pd.Series or array-like
        Test returns. If pd.Series, index is used as dates.

    model_names : list or None
        Models to plot. If None, plots all models.

    level_idx : int
        0 for 95% VaR, 1 for 99% VaR.

    confidences : tuple
        Confidence levels corresponding to VaR rows.

    figsize : tuple
        Figure size.
    """
    test = np.asarray(logret_test).ravel()

    if hasattr(logret_test, "index"):
        dates = logret_test.index
    else:
        dates = np.arange(len(test))

    if model_names is None:
        model_names = list(VaR_results.keys())

    VaR_level = int(confidences[level_idx] * 100)

    for model_name in model_names:
        VaR_path = np.asarray(VaR_results[model_name][level_idx]).ravel()
        breach_mask = np.asarray(breach_results[model_name][level_idx]).astype(bool)

        plt.figure(figsize=figsize)

        plt.plot(dates, test, label="Return")
        plt.plot(dates, -VaR_path, label=f"{VaR_level}% VaR threshold")

        plt.scatter(
            np.asarray(dates)[breach_mask],
            test[breach_mask],
            label="VaR breaches",
            zorder=3,
            color="red"
        )

        plt.axhline(0, linewidth=0.8)
        plt.legend()
        plt.title(f"{model_name}: {VaR_level}% VaR breaches")
        plt.xlabel("Date")
        plt.ylabel("Return (%)")
        plt.show()