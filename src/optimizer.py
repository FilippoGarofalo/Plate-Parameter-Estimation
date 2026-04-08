import torch.optim as optim

from model import DifferentiableModalPlate


def get_optimizer(model):
    # Initialize Adam Optimizer
    # We use custom learning rates
    '''
    optimizer = optim.Adam([
        # Ly, plate's length in y-direction
        {'params': [model.Ly_raw], 'lr': 0.005},
        
        # Material properties (mu, D/mu, T0/mu)
        {'params': [model.mu_raw, model.D_over_mu_raw, model.T0_over_mu_raw], 'lr': 0.005},
        
        # Mic placement (xo, yo)
        {'params': [model.xo_raw, model.yo_raw], 'lr': 0.02}
    ])
    '''
    # After normalizing raw params to O(1), all material params share the same LR.
    # Geometric params (xo, yo) use a slightly higher LR, Ly slightly lower.
    optimizer = optim.Adam([
        {'params': [model.Ly_raw], 'lr': 0.005},
        {'params': [model.xo_raw, model.yo_raw], 'lr': 0.02},
        {'params': [model.mu_raw, model.D_over_mu_raw, model.T0_over_mu_raw], 'lr': 0.01},
    ])
    
    return optimizer

