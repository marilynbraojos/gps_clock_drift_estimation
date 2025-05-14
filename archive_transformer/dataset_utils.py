import torch 
import numpy as np 
from torch.utils.data import TensorDataset 

def create_dataset(broadcast_clock_bias, correction_value, timesteps, corr_timestep):
    """
    Create input-output dataset
    
    Args: 
    broadcast_clock_bias: array of broadcast clock bias values
    correction_value: array of corresponding correction values
    """
    
    x = []  # Input
    y = []  # Labels

    # Generate history and correction samples
    for i in range(timesteps, len(broadcast_clock_bias) - corr_timestep):
        history = broadcast_clock_bias[i - timesteps:i]
        correction = correction_value[i - 1:i - 1 + corr_timestep]
        x.append(history)
        y.append(correction)

    # Convert to NumPy arrays
    x = np.array(x)
    y = np.array(y)

    # Convert to PyTorch tensors
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    # Create TensorDataset
    dataset = TensorDataset(x, y)

    return x, y, dataset