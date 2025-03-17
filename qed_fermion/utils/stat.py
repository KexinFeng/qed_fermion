import numpy as np
from scipy import stats
import torch

import matlab.engine

def error_mean(array, axis=0):
    """
    Calculate the error mean along a specified axis.
    The operation is: square -> mean -> root
    """
    squared = np.square(array)
    mean_squared = np.mean(squared, axis=axis)
    root_mean_squared = np.sqrt(mean_squared)
    return root_mean_squared

def init_convex_seq_estimator(array):
    """
    This function takes a numpy array as input, flattens all axes except the first,
    passes it to a MATLAB function to process the data, and then fetches the result
    from MATLAB and reshapes it back to the original shape.

    Estimator is applied on the first axis of the input array.
    """
    matlab_function_path = '/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/utils/init_seq_matlab'
    eng = matlab.engine.start_matlab()
    eng.addpath(matlab_function_path)
    original_shape = array.shape
    flattened_array = array.reshape(original_shape[0], -1)
    matlab_array = matlab.double(flattened_array.tolist())
    result = eng.initseq_matlab_vec(matlab_array)
    eng.quit()
    result_array = np.array(result)
    result_array = result_array.reshape(*original_shape[1:])
    return result_array**(1/2)


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
