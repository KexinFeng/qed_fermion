import gc
import torch
import inspect

if torch.cuda.is_available():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

def device_mem():
    if torch.cuda.is_available():
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return f"NVML Used: {info.used / 1024**2:.2f} MB", info.used / 1024**2
    else:
        return "no cuda", 0

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

def tensor_memory_MB(tensor, device='cpu'):
    if device is None:
        return tensor.element_size() * tensor.numel() / 1024**2
    elif device == 'cpu':
        return tensor.element_size() * tensor.numel() / 1024**2 if not tensor.is_cuda else 0
    else:
        return tensor.element_size() * tensor.numel() / 1024**2 if tensor.is_cuda else 0


def report_tensor_memory():
    tensor_info = []
    total = 0

    # Get current frame and local variables from calling scope
    current_frame = inspect.currentframe()
    outer_frames = inspect.getouterframes(current_frame)
    caller_locals = outer_frames[1].frame.f_locals

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                mem_MB = tensor_memory_MB(obj)

                # Try to find variable name from local scope
                var_names = [name for name, val in caller_locals.items() if val is obj]
                var_name = var_names[0] if var_names else "<unnamed>"

                tensor_info.append((mem_MB, obj, var_name))
                total += mem_MB
        except Exception:
            pass

    # Sort by memory in descending order
    tensor_info.sort(key=lambda x: x[0], reverse=True)

    for mem_MB, tensor, var_name in tensor_info:
        if tensor.is_cuda: continue
        print(f"{var_name}: {str(tensor.size())} | {tensor.dtype} | {mem_MB:.2f} MB")

    print(f"Total memory used by tensors: {total:.2f} MB")


def bond_corr(BB_r_mean, B_r_mean, bb_r_std=None, b_r_std=None):
    # BB_r_mean: [Ly, Lx], mean over configurations
    # B_r_mean: [Ly, Lx], mean over configurations
    # bb_r_std: [Ly, Lx], standard deviation of BB_r over configurations
    # b_r_std: [Ly, Lx], standard deviation of B_r over configurations
    Ly, Lx = BB_r_mean.shape[-2:]
    vi = B_r_mean   
    v_F_neg_k = torch.fft.ifftn(vi, (Ly, Lx), norm="backward")
    v_F = torch.fft.fftn(vi, (Ly, Lx), norm="forward")
    v_bg = torch.fft.ifftn(v_F_neg_k * v_F, (Ly, Lx), norm="forward")  # [Ly, Lx]
    bond_corr = BB_r_mean - 4 * v_bg.real
    
    if bb_r_std is None:
        return bond_corr

    # Error propagation for bond_corr
    # The error in bond_corr comes from:
    # 1. Error in BB_r_mean (bb_r_std)
    # 2. Error in v_bg, which depends on error in B_r_mean (b_r_std)
    
    # Error in v_bg: since v_bg = IFFT(FFT(B_r_mean) * IFFT(B_r_mean))
    # The error propagates through the FFT operations
    # For FFT operations, errors are generally preserved in magnitude
    # The error in v_bg is approximately the error in B_r_mean squared
    # because we're multiplying B_r_mean with itself in Fourier space
    
    # Error in v_bg (approximate)
    v_bg_error = torch.fft.ifftn(
        torch.fft.fftn(b_r_std, (Ly, Lx), norm="forward") * 
        torch.fft.ifftn(b_r_std, (Ly, Lx), norm="backward"), 
        (Ly, Lx), norm="forward"
    )
    
    # Error in bond_corr: sqrt(bb_r_std^2 + (4 * v_bg_error)^2)
    bond_corr_error = torch.sqrt(bb_r_std**2 + (4 * v_bg_error)**2)
    
    return bond_corr, bond_corr_error # Delta: [Ly, Lx], Error: [Ly, Lx]