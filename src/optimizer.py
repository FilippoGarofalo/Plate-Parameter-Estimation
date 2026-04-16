import torch.optim as optim

from model import DifferentiableModalPlate


def get_optimizer(model, lr: float = 0.01):
    # Single LR across all parameters.
    return optim.Adam(model.parameters(), lr=lr)

