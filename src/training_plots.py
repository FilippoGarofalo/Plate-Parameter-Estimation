import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_progress(checkpoint_path="target/train_progress.npz"):
    """
    Load training progress from checkpoint and create plots for each learnable parameter.
    Each plot shows the parameter error (absolute difference from target) over training iterations.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return

    # Load the checkpoint data
    data = np.load(checkpoint_path)

    iterations = data['iteration']
    parameters = {
        'mu': data['mu'],
        'D_over_mu': data['D_over_mu'],
        'T0_over_mu': data['T0_over_mu'],
        'Ly': data['Ly'],
        'xo': data['xo'],
        'yo': data['yo']
    }

    loss = data['loss']

    # Calculate target values (same as in train.py)
    Lx = 0.5
    Ly = 1.1
    h = 0.001
    T0 = 0.01
    rho = 2430.0
    E = 6.7e10
    nu = 0.25

    target_mu = rho * h
    target_D = (E * h**3) / (12 * (1 - nu**2))
    target_D_mu = target_D / target_mu
    target_T0_mu = T0 / target_mu
    target_Ly = Ly
    target_xo = 0.61 * Lx
    target_yo = 0.61 * Ly

    targets = {
        'mu': target_mu,
        'D_over_mu': target_D_mu,
        'T0_over_mu': target_T0_mu,
        'Ly': target_Ly,
        'xo': target_xo,
        'yo': target_yo
    }

    # Create subplots: 2 rows, 3 columns for 6 parameters
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    param_names = list(parameters.keys())

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        values = parameters[param_name]
        target = targets[param_name]

        # Calculate absolute error from target
        errors = np.abs(values - target)

        # Plot parameter error vs iteration
        ax.plot(iterations, errors, 'r-', linewidth=2, label=f'{param_name} error')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'{param_name} Error')
        ax.set_title(f'{param_name} Error Evolution (Target: {target:.6f})')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for error to show improvement better

        # Add final error annotation
        final_error = errors[-1]
        ax.annotate(f'Final Error: {final_error:.2e}',
                   xy=(iterations[-1], final_error),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                   fontsize=9)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('target/training_parameter_errors.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Also create a separate loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss, 'b-', linewidth=2, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Evolution')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for loss if it decreases exponentially
    plt.savefig('target/training_loss_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Plots saved to target/ directory:")
    print("- training_parameter_errors.png")
    print("- training_loss_progress.png")
    print("\nTarget values:")
    for param, target in targets.items():
        print(f"- {param}: {target:.6f}")

if __name__ == "__main__":
    plot_training_progress()
