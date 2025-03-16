import numpy as np
from scipy import stats
import torch



def std_root_n(array, axis=None, unbiased=True, lag_sum=1):
    std = np.std(array, axis=axis, ddof=1 if unbiased else 0)
    n = array.shape[axis] if axis is not None else array.size
    return std / np.sqrt(n / lag_sum)


def t_based_error(data, confidence=0.95, axis=0):
    """
    Calculate confidence interval using Student's t-distribution along a specified axis
    Returns: margin_of_error along the specified axis
    """
    n = data.shape[axis]
    if n < 2:
        raise ValueError("Sample size must be at least 2")

    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis, ddof=1)  # Sample standard deviation
    sem = std / np.sqrt(n)  # Standard error of the mean

    # Get t-critical value for (1 - α/2) confidence level
    alpha = 1 - confidence
    df = n - 1  # Degrees of freedom
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    error = t_crit * sem

    # # Check
    # ci_low, ci_high = stats.t.interval(
    #     confidence=0.95,
    #     df=len(data)-1,
    #     loc=np.mean(data),
    #     scale=stats.sem(data)
    # )
    # assert abs(ci_high - ci_low - 2 * error) < 1e-3

    return error
