import torch

def ravel_multi_index(multi_index, shape):
    """
    multi_index: tuple of 1D tensors, as from torch.unravel_index
    shape: tuple/list of ints
    Returns: 1D tensor of flat indices
    """
    flat_index = torch.zeros_like(multi_index[0])
    stride = 1
    for i in reversed(range(len(shape))):
        flat_index += multi_index[i] * stride
        stride *= shape[i]
    return flat_index