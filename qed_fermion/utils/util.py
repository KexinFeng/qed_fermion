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

def unravel_index(flat_index, shape):
    """
    flat_index: 1D tensor of flat indices
    shape: tuple/list of ints
    Returns: tuple of 1D tensors, as from torch.unravel_index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(flat_index % dim)
        flat_index //= dim
    return tuple(reversed(multi_index))
