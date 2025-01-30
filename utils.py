import numpy as np
import torch


def cosine_schedule(beta_init, beta_final, steps, s=0.008):
    """
    Cosine schedule for diffusion steps.

    Parameters:
    beta_init (float): Initial beta value.
    beta_final (float): Final beta value.
    steps (int): Total number of time steps.
    s (float): Small constant to avoid division by zero (default is 0.008).

    Returns:
    torch.Tensor: Scheduled beta values for each time step.
    """
    t = np.arange(steps)
    cosine_values = np.cos((t / steps + s) / (1 + s) * np.pi / 2) ** 2
    return torch.tensor(
        beta_init + (beta_final - beta_init) * (1 - cosine_values), dtype=torch.float32
    )
