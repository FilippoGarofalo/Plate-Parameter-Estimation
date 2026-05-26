import argparse
import gc
import os
import torch

from train import train_on_target


def build_ir_path(base_dir, ir_index):
    filename = f"random_IR_{ir_index:04d}.npz"
    return os.path.join(base_dir, filename)


def main():
    parser = argparse.ArgumentParser(description="Batch train on IR .npz files")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "target",
            "2026-DATASET-STRIPPED",
        ),
        help="Directory containing random_IR_XXXX.npz files",
    )
    parser.add_argument("--start", type=int, default=1, help="Start IR index")
    parser.add_argument("--end", type=int, default=16, help="End IR index")
    parser.add_argument("--print-every", type=int, default=100, help="Log every N iterations")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument(
        "--progress-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "target"),
        help="Directory to save per-IR progress files",
    )
    args = parser.parse_args()

    results = []
    base_dir = os.path.abspath(args.base_dir)
    progress_dir = os.path.abspath(args.progress_dir)

    for ir_index in range(args.start, args.end + 1):
        ir_path = build_ir_path(base_dir, ir_index)
        if not os.path.exists(ir_path):
            print(f"[warn] Missing IR file: {ir_path}")
            continue

        print(f"\n=== IR {ir_index:04d} | {ir_path} ===")
        progress_path = os.path.join(progress_dir, f"train_progress_IR_{ir_index:04d}.npz")

        result = train_on_target(
            ir_path,
            num_iterations=args.num_iterations,
            print_every=args.print_every,
            progress_path=progress_path,
        )

        results.append((ir_index, result))

        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== FINAL SUMMARY ===")
    for ir_index, result in results:
        params = result["params"]
        loss_val = result["loss"]
        loss_str = f"{loss_val:.2e}" if loss_val < 1e-3 else f"{loss_val:.6f}"
        print(f"IR {ir_index:04d} | Loss: {loss_str}")
        print(
            f"mu: {params['mu']:.6f} | D/mu: {params['D_over_mu']:.6f} | "
            f"T0/mu: {params['T0_over_mu']:.6f} | Ly: {params['Ly']:.4f} | "
            f"xo: {params['xo']:.4f} | yo: {params['yo']:.4f}"
        )
        print("-" * 60)


if __name__ == "__main__":
    main()
