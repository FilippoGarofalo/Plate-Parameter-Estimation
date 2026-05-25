import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from pathlib import Path

def estimate_t60(edc_db, sr):
    """Estimates T60 based on the time it takes to drop from -5dB to -25dB (T20 extrapolation)."""
    try:
        t_minus_5 = np.where(edc_db <= -5)[0][0]
        t_minus_25 = np.where(edc_db <= -25)[0][0]
        t20_samples = t_minus_25 - t_minus_5
        t60_sec = (t20_samples * 3) / sr
        return t60_sec
    except IndexError:
        return 0.0

def main():
    sr = 44100
    
    # Automatically generate the list of 16 filenames
    # zfill(4) ensures numbers are padded with zeros (e.g., 1 -> '0001')
    file_names = [f"random_IR_{str(i).zfill(4)}.npz" for i in range(1, 17)]
    
    # Robust path resolution: finds the 'target' folder one level above this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    target_dir = project_root / "target/2026-DATASET-STRIPPED"

    # Make the figure slightly wider to accommodate the legend on the side
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Comparative Analysis of Target IRs (0001 - 0016)", fontsize=16)

    # Use a colormap with enough distinct colors for 16 files
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(file_names))]

    print(f"{'Filename':<30} | {'Fund. Freq (Hz)':<15} | {'T60 (sec)':<10}")
    print("-" * 60)

    loaded_any_files = False
    time_axis = None

    for idx, fname in enumerate(file_names):
        filepath = target_dir / fname
        
        if not filepath.exists():
            print(f"Could not find: {filepath}")
            continue

        try:
            data = np.load(filepath)
            array_key = list(data.keys())[0] 
            ir = data[array_key].flatten()
            
            # --- CRITICAL FIX: Peak Normalization ---
            # Brings the signal to 0 dBFS so peaks can be detected
            ir = ir / (np.max(np.abs(ir)) + 1e-12)
            
            loaded_any_files = True
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        # ---------------------------------------------------------
        # 1. Frequency Analysis
        # ---------------------------------------------------------
        nfft = 16384
        f, Pxx = signal.welch(ir, sr, nperseg=nfft)
        Pxx_db = 10 * np.log10(Pxx + 1e-12)
        
        ax1.plot(f, Pxx_db, label=fname, color=colors[idx], alpha=0.8)
        
        peaks, _ = signal.find_peaks(Pxx_db, prominence=10, distance=50)
        fund_freq = f[peaks[0]] if len(peaks) > 0 else 0.0

        # ---------------------------------------------------------
        # 2. Schroeder Energy Decay Curve
        # ---------------------------------------------------------
        energy = ir ** 2
        edc = np.cumsum(energy[::-1])[::-1]
        edc_db = 10 * np.log10(edc / (edc[0] + 1e-12) + 1e-12)
        time_axis = np.linspace(0, len(ir)/sr, len(ir))
        
        ax2.plot(time_axis, edc_db, label=fname, color=colors[idx], alpha=0.8)
        
        t60 = estimate_t60(edc_db, sr)

        print(f"{fname:<30} | {fund_freq:<15.2f} | {t60:<10.2f}")

    # Safety check: prevent plotting crash if no files loaded
    if not loaded_any_files:
        print("\n[!] No files were successfully loaded. Exiting before plotting.")
        return

    # Format Plots
    ax1.set_xlim(0, 2000)
    # Automatically adjust Y-limits based on the normalized data
    ax1.set_ylim(-80, 5) 
    ax1.set_title("Magnitude Spectrum (Low Frequency Detail)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, alpha=0.3)
    
    # Place legend outside the plot so it doesn't cover the data
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small', ncol=1)

    ax2.set_xlim(0, time_axis[-1] if time_axis is not None else 1.0)
    ax2.set_ylim(-60, 0)
    ax2.set_title("Schroeder Energy Decay Curve")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Energy (dB)")
    ax2.grid(True, alpha=0.3)
    
    # Place legend outside the plot
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small', ncol=1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()