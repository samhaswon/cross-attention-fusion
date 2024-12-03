"""
Determines the appropriate device for the model to be loaded to.
"""
import torch


def get_device():
    """
    Determines the device to run the model on (GPU/CPU).
    :returns: Device type ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        print("NVIDIA CUDA acceleration enabled")
        torch.multiprocessing.set_start_method("spawn")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Apple Metal Performance Shaders acceleration enabled")
        torch.multiprocessing.set_start_method("fork")
        return torch.device("mps")
    print("No GPU acceleration available")
    return torch.device("cpu")
