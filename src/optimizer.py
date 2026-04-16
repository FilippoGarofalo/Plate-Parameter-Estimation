import torch.optim as optim

from model import DifferentiableModalPlate


def get_optimizer(model, lr: float = 0.01):
    # Single LR across all parameters.
    # Works because model.py applies _SCALE constants to softplus outputs so that
    # the normalized Jacobian ∂L/∂raw is O(sigmoid(raw)) for every parameter.
    return optim.Adam(model.parameters(), lr=lr)

