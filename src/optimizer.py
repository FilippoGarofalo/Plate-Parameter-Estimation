import torch.optim as optim

def get_optimizer(model, lr: float = 0.1):
    slow_params = []
    fast_params = []
    
    # Iterate through all parameters and their actual variable names
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Group Ly and T0_over_mu together
        if 'Ly' in name or 'T0_over_mu' in name:
            slow_params.append(param)
        else:
            # Everything else (mu, D_over_mu, xo, yo)
            fast_params.append(param)
            
    return optim.Adam([
        {'params': fast_params, 'lr': 0.01},    # Default to 0.1 for the rest
        {'params': slow_params, 'lr': 0.1}   # Lock Ly and T0_over_mu to 0.01
    ])