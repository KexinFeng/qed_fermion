
def clear_mat(mat, threshold=1e-5):
    """
    Sets entries of the matrix whose absolute value is below `threshold` to zero.
    
    Args:
        mat (torch.Tensor): Input matrix.
        threshold (float, optional): Threshold value. Default is 1e-3.
    
    Returns:
        torch.Tensor: Thresholded matrix.
    """
    return mat * (mat.abs() >= threshold)
