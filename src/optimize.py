import torch
import torch.optim as optim
import time
from src.model import DifferentiableModalPlate
from src.loss import MultiScaleSpectralLoss
from src.audio_utils import load_target_audio

def main():
    # ---------------------------------------------------------
    # 1. SETUP & HYPERPARAMETERS
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Path to the challenge target file (ensure this exists or use a dummy)
    target_audio_path = "dataset/TargetDataset/target_ir_01.wav" 
    
    learning_rate = 0.05
    num_iterations = 500
    sample_rate = 44100
    
    # ---------------------------------------------------------
    # 2. INITIALIZE MODULES
    # ---------------------------------------------------------
    # Load target audio and determine how many samples to synthesize
    try:
        target_ir = load_target_audio(target_audio_path, target_sr=sample_rate, device=device)
        num_samples = len(target_ir)
        print(f"Loaded target IR: {num_samples} samples.")
    except Exception as e:
        print(f"Failed to load target audio: {e}. Please ensure the path is correct.")
        return

    model = DifferentiableModalPlate(sample_rate=sample_rate).to(device)
    criterion = MultiScaleSpectralLoss().to(device)
    
    # We use Adam, which is highly effective for DDSP topologies
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ---------------------------------------------------------
    # 3. OPTIMIZATION LOOP
    # ---------------------------------------------------------
    print("\nStarting Optimization...")
    start_time = time.time()
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward Pass: Physics -> Audio
        pred_ir = model(num_samples)
        
        # Compute Loss: Audio vs Target
        loss = criterion(pred_ir, target_ir)
        
        # Backward Pass: Calculate gradients
        loss.backward()
        
        # Update physical parameters
        optimizer.step()
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration:04d} | Loss: {loss.item():.6f}")

    total_time = time.time() - start_time
    print(f"\nOptimization complete in {total_time:.2f} seconds.")

    # ---------------------------------------------------------
    # 4. EXTRACT RESULTS FOR THE CHALLENGE SUBMISSION
    # ---------------------------------------------------------
    # We detach the final physical parameters from the computation graph
    mu, D_over_mu, T0_over_mu, Ly, xo, yo = [
        p.detach().cpu().item() for p in model.get_physical_parameters()
    ]
    
    print("\n=== FINAL ESTIMATED PARAMETERS ===")
    print(f"mu         := rho/h     = {mu:.6f}")
    print(f"D/mu       := Rigidity  = {D_over_mu:.6f}")
    print(f"T0/mu      := Tension   = {T0_over_mu:.6f}")
    print(f"Ly         := Height    = {Ly:.4f} m")
    print(f"xo         := Output X  = {xo:.4f} m")
    print(f"yo         := Output Y  = {yo:.4f} m")
    print("==================================")
    
    # (Here you would write these values, along with total_time and num_iterations, 
    # to the CSV format specified by the DAFx challenge guidelines.)

if __name__ == "__main__":
    main()