import torch
import time
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from model import DifferentiableModalPlate
from loss import Loss
from loss2 import MSELoss
from utils import load_challenge_npz
from optimizer import get_optimizer


def get_dsp_scheduler(optimizer, warmup_steps=100, decay_steps=1000, min_lr_ratio=0.01):

    def lr_lambda(current_step):
        #Phase 1
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
            
        #Phase 2:
        progress = float(current_step - warmup_steps) / float(max(1, decay_steps))
        progress = min(1.0, progress) 
            
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

def main():
    # 1. SETUP & HYPERPARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    target_npz_path = "target/ground_truth_test_1.npz"
    sample_rate = 44100
    num_iterations = 2000
    LR = 0.01
    dtype = torch.float32

    target_ir = load_challenge_npz(target_npz_path, device=device, dtype=dtype)

    duration = len(target_ir) / sample_rate
    print(f"Target IR loaded: {len(target_ir)} samples ({duration:.2f} seconds)")

    # 2. INITIALIZE MODULES
    model = DifferentiableModalPlate(sample_rate=sample_rate, plate_params=None, dtype=dtype).to(device)
    criterion = Loss(
        mse_weight=0.0,
        stft_weight=1.0,
        energy_weight=0.0,
        fft_sizes=[64, 256, 1024, 4096]
       ).to(device)

    #model.Ly_raw.requires_grad = False
    #model.xo_raw.requires_grad = False
    #model.yo_raw.requires_grad = False

    active_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = get_optimizer(active_params ,lr=LR)
    
    # OPTIMIZATION: Precompute target STFT once (cached for all iterations)
    criterion.precompute_target_stft(target_ir)
    #criterion = MSELoss().to(device)
    scheduler = get_dsp_scheduler(optimizer, warmup_steps=100, decay_steps=num_iterations-100)
    progress = {'iteration': [], 'loss': [], 'mu': [], 'D_over_mu': [], 'T0_over_mu': [], 'Ly': [], 'xo': [], 'yo': []}

    # 3. OPTIMIZATION LOOP
    print("\nStarting Optimization")
    start_time = time.time()
    for iteration in range(num_iterations):
        # Step 1: Clear the gradients
        optimizer.zero_grad()

        # Step 2: Forward Pass
        if iteration == 0: 
            print(" [diag] forward...", flush=True)
        pred_ir = model(duration=duration, normalize=False, velCalc=False)
        if iteration == 0: 
            print(" [diag] loss...", flush=True)
        loss = criterion(pred_ir, target_ir)

        # Step 4: Backward Pass
        if iteration == 0: 
            print(f" [diag] loss={loss.item():.6f} backward...", flush=True)
        loss.backward()

        # Step 5: Gradient Clipping (Crucial for stability with Adam and physical parameters)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if iteration == 0:
            grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
            print(f" [diag] grad norms: {grad_norms}", flush=True)

        # Step 6: Update Parameters
        optimizer.step()
        scheduler.step()  # Update the learning rate
        optimizer.zero_grad()

        # Step 7: Print logs and parameter progress
        if iteration % 10 == 0 or iteration == num_iterations - 1:
            # Safely extract the current bounded physical values
            mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
            p.detach().cpu().item() for p in model.get_physical_parameters()
            ]
            print(f"Iteration {iteration:04d} | Loss: {loss.item():.6f}")
            print(f"Ly: {Ly:.4f}m | xo: {xo:.4f}m | yo: {yo:.4f}m | "
            f"mu: {mu:.4f} | D/mu: {D_over_mu:.6f} | T0/mu: {T0_over_mu:.6f}")
            print("-" * 60)

            progress['iteration'].append(iteration)
            progress['loss'].append(loss.item())
            progress['mu'].append(mu)
            progress['D_over_mu'].append(D_over_mu)
            progress['T0_over_mu'].append(T0_over_mu)
            progress['Ly'].append(Ly)
            progress['xo'].append(xo)
            progress['yo'].append(yo)

    total_time = time.time() - start_time
    print(f"\nOptimization complete in {total_time:.2f} seconds.")

    np.savez('target/train_progress.npz', **{k: np.array(v) for k, v in progress.items()})
    print("Training progress saved to target/train_progress.npz")

    # 4. RESULTS
    # ---------------------------------------------------------
    mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
    p.detach().cpu().item() for p in model.get_physical_parameters()
    ]
    print("\n=== FINAL ESTIMATED PARAMETERS ===")
    print(f"mu := {mu:.6f}")
    print(f"D/mu := {D_over_mu:.6f}")
    print(f"T0/mu := {T0_over_mu:.6f}")
    print(f"Ly := {Ly:.4f} m")
    print(f"xo := {xo:.4f} m")
    print(f"yo := {yo:.4f} m")
    print("==================================")

if __name__ == "__main__":
    main()