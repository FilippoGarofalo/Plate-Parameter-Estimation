# Parameter Mapping Memo: Task A (Modal Plate)

This document tracks the relationship between the physical math variables described in the challenge specifications and their corresponding PyTorch tensor representations in the `DifferentiableModalPlate` module.

## 🎯 1. Learnable Parameters (The Optimization Targets)
These are the 6 parameters we are trying to estimate. In PyTorch, they are initialized as unconstrained latent variables (`_raw`) centered around `0.0` and mapped to their strict physical bounds during the `forward()` pass.

| Math Symbol | Description | PyTorch Latent Variable | PyTorch Physical Variable | Transformation / Bounding Function | Physical Bounds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\mu$ | Density-thickness ratio ($\rho/h$) | `self.mu_raw` | `mu` | `F.softplus(raw) + 1e-4` | $> 0$ (Strictly positive) |
| $D/\mu$ | Rigidity ratio | `self.D_over_mu_raw` | `D_over_mu` | `F.softplus(raw) + 1e-4` | $> 0$ (Strictly positive) |
| $T_0/\mu$ | Tension ratio | `self.T0_over_mu_raw`| `T0_over_mu` | `F.softplus(raw) + 1e-4` | $> 0$ (Strictly positive) |
| $L_y$ | Plate height | `self.Ly_raw` | `Ly` | `1.1 + (2.9) * sigmoid(raw)` | **1.1 m** to **4.0 m** |
| $x_o$ | Output transducer X-position | `self.xo_raw` | `xo` | `0.49*Lx + (0.51*Lx) * sigmoid(raw)` | **0.49 $\cdot L_x$** to **$L_x$** |
| $y_o$ | Output transducer Y-position | `self.yo_raw` | `yo` | `0.51*Ly + (0.49*Ly) * sigmoid(raw)` | **0.51 $\cdot L_y$** to **$L_y$** |

*Note: The unconstrained `_raw` variables are registered as `nn.Parameter` so the Adam optimizer can freely update them from $-\infty$ to $+\infty$ without breaking the physics.*

---

## 🔒 2. Fixed Physical Parameters (Challenge Constants)
These values are explicitly fixed by the DAFx challenge. They dictate the size, damping, and input strike location of the plate. They are registered as PyTorch `buffers` so they automatically move to the GPU but do not receive gradients.

| Math Symbol | Description | PyTorch Variable | Fixed Value | Status |
| :--- | :--- | :--- | :--- | :--- |
| $L_x$ | Plate width | `self.Lx` | **1.0** m | `register_buffer` |
| $\nu$ | Poisson’s ratio | `self.nu` | **0.25** | `register_buffer` |
| $\tau_0$ | Decay time at DC (0 Hz) | `self.tau_0` | **6.0** s | `register_buffer` |
| $\tau_1$ | Decay time at $f_1$ | `self.tau_1` | **2.0** s | `register_buffer` |
| $f_1$ | Reference loss frequency | `self.loss_f1` | **500.0** Hz | `register_buffer` |
| $x_i$ | Input position X | `xi` (Computed in `forward`)| **0.335 $\cdot L_x$** | Constant derived from $L_x$ |
| $y_i$ | Input position Y | `yi` (Computed in `forward`)| **0.467 $\cdot L_y$** | Dynamic dependency on learnable $L_y$ |

*Note: Because $y_i$ depends on the learnable parameter $L_y$, it cannot be pre-computed in `__init__`. It must be recalculated dynamically inside the `forward()` pass so gradients can flow properly.*

---

## ⚙️ 3. Internal DSP Constants (Implementation Details)
These govern the digital signal processing environment and the modal grid size.

| Variable Name | Description | Value | PyTorch Implementation |
| :--- | :--- | :--- | :--- |
| `sample_rate` | Global audio sampling rate | **44100** Hz | Passed to `__init__` |
| `fmax` | Maximum modal frequency synthesized | **10000.0** Hz | Float constant |
| `M_max`, `N_max` | Dimensions of the modal grid | **80 $\times$ 80** | Flattened into `m_vec`, `n_vec` buffers |
| `temp` | Sigmoid steepness for frequency masking | **100.0** | Local float in `forward()` |