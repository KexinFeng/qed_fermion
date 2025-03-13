import numpy as np
from scipy import stats
import torch


def std_root_n(tensor, dim=None, unbiased=True):
    std = torch.std(tensor, dim=dim, unbiased=unbiased)
    n = tensor.size(dim) if dim is not None else tensor.numel()
    return std / torch.sqrt(torch.tensor(n, dtype=torch.float32))


def t_based_error(data, confidence=0.95):
    """
    Calculate confidence interval using Student's t-distribution
    Returns: (mean, margin_of_error)
    """
    n = len(data)
    if n < 2:
        raise ValueError("Sample size must be at least 2")

    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
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
