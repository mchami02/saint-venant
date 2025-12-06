"""
Visualize FNO Spectral Representations

This script visualizes what happens inside the FNO at each Fourier layer:
1. The spectral representation (FFT) of the signal entering each layer
2. The learned spectral weights that modify each frequency
3. How the signal evolves through the network

The visualization shows:
- Input/Output comparison
- Spectral weights (what the network learned)
- Signal spectrum at each layer (min/max/mean across hidden channels)
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from operator_data_pipeline import GridDataset
from model import create_model
from numerical_methods import Godunov, Greenshields, LWRRiemannSolver
from nfv.initial_conditions import Riemann
from nfv.problem import Problem
from nfv.flows import Greenshield
from nfv.solvers import LaxHopf
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpectralCaptureHook:
    """Hook to capture spectral representations after FFT in each SpectralConv layer."""
    
    def __init__(self):
        self.spectral_inputs = []   # FFT of input to each layer
        self.spectral_outputs = []  # FFT of output from each layer (before IFFT)
        self.handles = []
    
    def clear(self):
        self.spectral_inputs = []
        self.spectral_outputs = []
    
    def hook_fn(self, module, input, output):
        """Capture both input and output spectral representations."""
        x = input[0]  # Input to SpectralConv
        
        # Replicate the FFT logic from SpectralConv
        order = len(module.n_modes)
        fft_dims = list(range(-order, 0))
        
        # FFT of input
        if module.complex_data:
            x_fft = torch.fft.fftn(x, norm=module.fft_norm, dim=fft_dims)
        else:
            x_fft = torch.fft.rfftn(x, norm=module.fft_norm, dim=fft_dims)
        
        # FFT of output (before the IFFT in forward)
        # We compute FFT of output to see what spectrum we're producing
        if module.complex_data:
            out_fft = torch.fft.fftn(output, norm=module.fft_norm, dim=fft_dims)
        else:
            out_fft = torch.fft.rfftn(output, norm=module.fft_norm, dim=fft_dims)
        
        self.spectral_inputs.append(torch.abs(x_fft).detach().cpu())
        self.spectral_outputs.append(torch.abs(out_fft).detach().cpu())
    
    def register_hooks(self, model):
        """Register forward hooks on all SpectralConv layers."""
        for i, conv in enumerate(model.fno_blocks.convs):
            handle = conv.register_forward_hook(self.hook_fn)
            self.handles.append(handle)
        print(f"Registered hooks on {len(self.handles)} SpectralConv layers")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []


def extract_spectral_weights(model):
    """Extract the learned spectral weights from each SpectralConv layer."""
    weights = []
    for conv in model.fno_blocks.convs:
        # Weight shape: (in_channels, out_channels, *n_modes) for dense
        # Get the weight tensor
        w = conv.weight
        if hasattr(w, 'to_tensor'):
            w = w.to_tensor()
        # Compute magnitude and average across in/out channels
        w_mag = torch.abs(w).detach().cpu()
        # Average across channel dimensions to get (freq_t, freq_x)
        w_mean = w_mag.mean(dim=(0, 1)).numpy()
        weights.append(w_mean)
    return weights


def plot_spectral_visualization(model, sample_input, sample_target, save_path="results/fno_spectral_viz.png"):
    """
    Create comprehensive visualization of FNO spectral representations.
    """
    model.eval()
    hook = SpectralCaptureHook()
    hook.register_hooks(model)
    
    # Forward pass to capture spectral representations
    with torch.no_grad():
        sample_input = sample_input.to(device)
        output = model(sample_input)
    
    # Get data
    spectral_inputs = hook.spectral_inputs
    spectral_outputs = hook.spectral_outputs
    n_layers = len(spectral_inputs)
    spectral_weights = extract_spectral_weights(model)
    
    hook.remove_hooks()
    
    # Create figure: 2 rows, n_layers + 1 columns
    # Row 0: Input/Output + Spectral weights for each layer
    # Row 1: GT/Error + Input spectrum (mean) for each layer
    n_cols = n_layers + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8))
    
    cmap_spatial = 'RdBu_r'
    cmap_spectral = 'viridis'
    cmap_weights = 'plasma'
    
    # ============================================
    # Column 0: Spatial domain comparison
    # ============================================
    
    input_np = sample_input[0, 0].cpu().numpy()
    output_np = output[0, 0].cpu().numpy()
    target_np = sample_target[0, 0].cpu().numpy()
    error_np = np.abs(output_np - target_np)
    
    # Row 0: Model Output
    vmax = max(abs(output_np.min()), abs(output_np.max()))
    im0 = axes[0, 0].imshow(output_np, aspect='auto', cmap=cmap_spatial, 
                             origin='lower', vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title('Model Output', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Space (x)')
    axes[0, 0].set_ylabel('Time (t)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Row 1: Absolute Error
    im1 = axes[1, 0].imshow(error_np, aspect='auto', cmap='Reds', origin='lower')
    axes[1, 0].set_title(f'|Error| (MSE={error_np.mean():.2e})', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Space (x)')
    axes[1, 0].set_ylabel('Time (t)')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # ============================================
    # Columns 1 to n_layers: Spectral info per layer
    # Each plot uses its OWN scale
    # ============================================
    
    for layer_idx in range(n_layers):
        col = layer_idx + 1
        
        # Row 0: Learned spectral weights - OWN SCALE
        w = spectral_weights[layer_idx]
        w_min = w.min() + 1e-6
        w_max = w.max()
        im_w = axes[0, col].imshow(w, aspect='auto', cmap=cmap_weights, 
                                    origin='lower', norm=LogNorm(vmin=w_min, vmax=w_max))
        axes[0, col].set_title(f'Layer {layer_idx}: Learned Weights', fontsize=12, fontweight='bold')
        axes[0, col].set_xlabel(f'Freq x (modes: {w.shape[1]})')
        axes[0, col].set_ylabel(f'Freq t (modes: {w.shape[0]})')
        plt.colorbar(im_w, ax=axes[0, col], fraction=0.046, pad=0.04)
        
        # Row 1: Input spectrum (mean across hidden channels) - OWN SCALE
        spec_in = spectral_inputs[layer_idx][0].numpy()  # (hidden, freq_t, freq_x)
        spec_mean = spec_in.mean(axis=0)
        spec_min = spec_mean.min() + 1e-10
        spec_max = spec_mean.max()
        
        im_s = axes[1, col].imshow(spec_mean, aspect='auto', cmap=cmap_spectral,
                                    origin='lower', norm=LogNorm(vmin=spec_min, vmax=spec_max))
        axes[1, col].set_title(f'Layer {layer_idx}: Input Spectrum', fontsize=12, fontweight='bold')
        axes[1, col].set_xlabel('Freq (x)')
        axes[1, col].set_ylabel('Freq (t)')
        plt.colorbar(im_s, ax=axes[1, col], fraction=0.046, pad=0.04)
    
    fig.suptitle('FNO Spectral Analysis: Learned Weights & Signal Spectrum per Layer', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Spectral visualization saved to {save_path}")
    
    # Print info
    print("\nSpectral weight shapes (what the network learned):")
    for i, w in enumerate(spectral_weights):
        print(f"  Layer {i}: {w.shape} (freq_t, freq_x)")
    
    print("\nInput spectrum shapes:")
    for i, s in enumerate(spectral_inputs):
        print(f"  Layer {i}: {s.shape} (batch, hidden_channels, freq_t, freq_x)")


def plot_detailed_spectrum(model, sample_input, sample_target, save_path="results/fno_spectrum_detailed.png"):
    """
    Create a more detailed view showing min/max/mean of spectrum at each layer.
    """
    model.eval()
    hook = SpectralCaptureHook()
    hook.register_hooks(model)
    
    with torch.no_grad():
        sample_input = sample_input.to(device)
        output = model(sample_input)
    
    spectral_inputs = hook.spectral_inputs
    n_layers = len(spectral_inputs)
    hook.remove_hooks()
    
    # 3 rows (min/max/mean), n_layers + 1 columns
    n_cols = n_layers + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 10))
    
    cmap_output = 'RdBu_r'
    cmap_spectral = 'magma'
    
    # Column 0: Spatial domain
    input_np = sample_input[0, 0].cpu().numpy()
    output_np = output[0, 0].cpu().numpy()
    target_np = sample_target[0, 0].cpu().numpy()
    
    vmax = max(abs(target_np.min()), abs(target_np.max()))
    
    axes[0, 0].imshow(input_np, aspect='auto', cmap='gray', origin='lower')
    axes[0, 0].set_title('Input (masked)', fontsize=11, fontweight='bold')
    
    axes[1, 0].imshow(output_np, aspect='auto', cmap=cmap_output, origin='lower', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('Model Output', fontsize=11, fontweight='bold')
    
    axes[2, 0].imshow(target_np, aspect='auto', cmap=cmap_output, origin='lower', vmin=-vmax, vmax=vmax)
    axes[2, 0].set_title('Ground Truth', fontsize=11, fontweight='bold')
    
    for ax in axes[:, 0]:
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('Time (t)')
    
    # Columns 1+: Spectral statistics - EACH WITH OWN SCALE
    for layer_idx, spectral in enumerate(spectral_inputs):
        col = layer_idx + 1
        spec_np = spectral[0].numpy()  # (hidden, freq_t, freq_x)
        
        spec_min_across_hidden = spec_np.min(axis=0)
        spec_max_across_hidden = spec_np.max(axis=0)
        spec_mean_across_hidden = spec_np.mean(axis=0)
        
        eps = 1e-10
        
        # Row 0: Min - OWN SCALE
        vmin0 = spec_min_across_hidden.min() + eps
        vmax0 = spec_min_across_hidden.max()
        im0 = axes[0, col].imshow(spec_min_across_hidden + eps, aspect='auto', cmap=cmap_spectral,
                                   origin='lower', norm=LogNorm(vmin=vmin0, vmax=vmax0))
        axes[0, col].set_title(f'Layer {layer_idx}: Min', fontsize=11, fontweight='bold')
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)
        
        # Row 1: Max - OWN SCALE
        vmin1 = spec_max_across_hidden.min() + eps
        vmax1 = spec_max_across_hidden.max()
        im1 = axes[1, col].imshow(spec_max_across_hidden + eps, aspect='auto', cmap=cmap_spectral,
                                   origin='lower', norm=LogNorm(vmin=vmin1, vmax=vmax1))
        axes[1, col].set_title(f'Layer {layer_idx}: Max', fontsize=11, fontweight='bold')
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
        
        # Row 2: Mean - OWN SCALE
        vmin2 = spec_mean_across_hidden.min() + eps
        vmax2 = spec_mean_across_hidden.max()
        im2 = axes[2, col].imshow(spec_mean_across_hidden + eps, aspect='auto', cmap=cmap_spectral,
                                   origin='lower', norm=LogNorm(vmin=vmin2, vmax=vmax2))
        axes[2, col].set_title(f'Layer {layer_idx}: Mean', fontsize=11, fontweight='bold')
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)
        
        for row in range(3):
            axes[row, col].set_xlabel('Freq (x)')
            axes[row, col].set_ylabel('Freq (t)')
    
    fig.suptitle('FNO Spectral Representation: Min/Max/Mean across Hidden Channels', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Detailed spectrum visualization saved to {save_path}")


def plot_before_after_spectrum(model, sample_input, sample_target, save_path="results/fno_before_after.png"):
    """
    Compare spectrum BEFORE and AFTER each Fourier layer.
    This shows exactly what each layer changes in the frequency domain.
    Each plot uses its own scale for maximum visibility.
    """
    model.eval()
    hook = SpectralCaptureHook()
    hook.register_hooks(model)
    
    with torch.no_grad():
        sample_input = sample_input.to(device)
        output = model(sample_input)
    
    spectral_inputs = hook.spectral_inputs
    spectral_outputs = hook.spectral_outputs
    n_layers = len(spectral_inputs)
    hook.remove_hooks()
    
    # Create figure: 4 rows x n_layers columns
    # Row 0: Input spectrum (before)
    # Row 1: Output spectrum (after)  
    # Row 2: Ratio (after/before) - what was amplified/suppressed
    # Row 3: Difference (after - before) - absolute change
    fig, axes = plt.subplots(4, n_layers, figsize=(5 * n_layers, 14))
    
    if n_layers == 1:
        axes = axes.reshape(-1, 1)
    
    cmap_spectrum = 'viridis'
    cmap_ratio = 'RdBu_r'  # Red = amplified, Blue = suppressed
    cmap_diff = 'PiYG'     # Green = increased, Pink = decreased
    
    for layer_idx in range(n_layers):
        # Get mean spectrum across hidden channels
        spec_in = spectral_inputs[layer_idx][0].numpy().mean(axis=0)   # (freq_t, freq_x)
        spec_out = spectral_outputs[layer_idx][0].numpy().mean(axis=0)
        
        eps = 1e-10
        
        # Row 0: Input spectrum (before layer) - OWN SCALE
        spec_in_min = spec_in.min() + eps
        spec_in_max = spec_in.max()
        im0 = axes[0, layer_idx].imshow(
            spec_in + eps, aspect='auto', cmap=cmap_spectrum,
            origin='lower', norm=LogNorm(vmin=spec_in_min, vmax=spec_in_max)
        )
        axes[0, layer_idx].set_title(f'Layer {layer_idx}: BEFORE\n(Input Spectrum)', 
                                      fontsize=11, fontweight='bold')
        axes[0, layer_idx].set_ylabel('Freq (t)')
        plt.colorbar(im0, ax=axes[0, layer_idx], fraction=0.046, pad=0.04)
        
        # Row 1: Output spectrum (after layer) - OWN SCALE
        spec_out_min = spec_out.min() + eps
        spec_out_max = spec_out.max()
        im1 = axes[1, layer_idx].imshow(
            spec_out + eps, aspect='auto', cmap=cmap_spectrum,
            origin='lower', norm=LogNorm(vmin=spec_out_min, vmax=spec_out_max)
        )
        axes[1, layer_idx].set_title(f'Layer {layer_idx}: AFTER\n(Output Spectrum)', 
                                      fontsize=11, fontweight='bold')
        axes[1, layer_idx].set_ylabel('Freq (t)')
        plt.colorbar(im1, ax=axes[1, layer_idx], fraction=0.046, pad=0.04)
        
        # Row 2: Ratio (after/before) - log scale centered at 0 (=ratio of 1)
        ratio = (spec_out + eps) / (spec_in + eps)
        log_ratio = np.log10(ratio)
        # Use percentile to avoid outliers dominating the scale
        vmax_ratio = np.percentile(np.abs(log_ratio), 99)
        vmax_ratio = max(vmax_ratio, 0.1)  # Ensure minimum range
        
        im2 = axes[2, layer_idx].imshow(
            log_ratio, aspect='auto', cmap=cmap_ratio,
            origin='lower', vmin=-vmax_ratio, vmax=vmax_ratio
        )
        axes[2, layer_idx].set_title(f'Layer {layer_idx}: RATIO\n(log₁₀(after/before))', 
                                      fontsize=11, fontweight='bold')
        axes[2, layer_idx].set_ylabel('Freq (t)')
        cbar2 = plt.colorbar(im2, ax=axes[2, layer_idx], fraction=0.046, pad=0.04)
        cbar2.set_label('Red=amplified, Blue=suppressed')
        
        # Row 3: Absolute difference - OWN SCALE
        diff = spec_out - spec_in
        # Use percentile to set scale
        vmax_diff = np.percentile(np.abs(diff), 99)
        vmax_diff = max(vmax_diff, eps)  # Ensure minimum range
        
        im3 = axes[3, layer_idx].imshow(
            diff, aspect='auto', cmap=cmap_diff,
            origin='lower', vmin=-vmax_diff, vmax=vmax_diff
        )
        axes[3, layer_idx].set_title(f'Layer {layer_idx}: DIFFERENCE\n(after - before)', 
                                      fontsize=11, fontweight='bold')
        axes[3, layer_idx].set_xlabel('Freq (x)')
        axes[3, layer_idx].set_ylabel('Freq (t)')
        plt.colorbar(im3, ax=axes[3, layer_idx], fraction=0.046, pad=0.04)
    
    fig.suptitle('FNO Layer-by-Layer Spectral Transformation\n(How each Fourier layer modifies the frequency content)', 
                 fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Before/after spectrum visualization saved to {save_path}")
    
    # Print summary statistics
    print("\nLayer-by-layer spectral changes:")
    for layer_idx in range(n_layers):
        spec_in = spectral_inputs[layer_idx][0].numpy().mean(axis=0)
        spec_out = spectral_outputs[layer_idx][0].numpy().mean(axis=0)
        ratio = spec_out / (spec_in + 1e-10)
        print(f"  Layer {layer_idx}: max amplification={ratio.max():.2f}x, max suppression={ratio.min():.4f}x")


def plot_spectral_evolution(model, sample_input, sample_target, save_path="results/fno_spectral_evolution.png"):
    """
    Show how the full signal spectrum evolves from input to output through all layers.
    Single row showing: Input → Layer0 → Layer1 → Layer2 → Layer3 → Output
    Each plot has its own scale for better visibility.
    """
    model.eval()
    hook = SpectralCaptureHook()
    hook.register_hooks(model)
    
    with torch.no_grad():
        sample_input_dev = sample_input.to(device)
        output = model(sample_input_dev)
    
    spectral_inputs = hook.spectral_inputs
    n_layers = len(spectral_inputs)
    hook.remove_hooks()
    
    # Compute FFT of raw input and final output
    input_fft = torch.fft.rfftn(sample_input, dim=(-2, -1))
    output_fft = torch.fft.rfftn(output.cpu(), dim=(-2, -1))
    
    input_spec = torch.abs(input_fft[0, 0]).numpy()
    output_spec = torch.abs(output_fft[0, 0]).numpy()
    
    # Create figure: 1 row showing evolution
    n_cols = n_layers + 2  # Input + n_layers + Output
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    cmap = 'magma'
    eps = 1e-10
    
    # Column 0: Raw input spectrum - USE ITS OWN SCALE
    spec_min, spec_max = input_spec.min() + eps, input_spec.max()
    im0 = axes[0].imshow(input_spec + eps, aspect='auto', cmap=cmap, origin='lower',
                          norm=LogNorm(vmin=spec_min, vmax=spec_max))
    axes[0].set_title('Raw Input\n(before lifting)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Freq (x)')
    axes[0].set_ylabel('Freq (t)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Columns 1 to n_layers: After each layer - EACH WITH OWN SCALE
    for layer_idx in range(n_layers):
        col = layer_idx + 1
        spec = spectral_inputs[layer_idx][0].numpy().mean(axis=0)
        spec_min, spec_max = spec.min() + eps, spec.max()
        
        im = axes[col].imshow(spec + eps, aspect='auto', cmap=cmap, origin='lower',
                               norm=LogNorm(vmin=spec_min, vmax=spec_max))
        axes[col].set_title(f'Before Layer {layer_idx}\n(hidden repr)', fontsize=11, fontweight='bold')
        axes[col].set_xlabel('Freq (x)')
        if col == 1:
            axes[col].set_ylabel('Freq (t)')
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
    
    # Last column: Final output spectrum - USE ITS OWN SCALE
    spec_min, spec_max = output_spec.min() + eps, output_spec.max()
    im_out = axes[-1].imshow(output_spec + eps, aspect='auto', cmap=cmap, origin='lower',
                              norm=LogNorm(vmin=spec_min, vmax=spec_max))
    axes[-1].set_title('Final Output\n(after projection)', fontsize=11, fontweight='bold')
    axes[-1].set_xlabel('Freq (x)')
    plt.colorbar(im_out, ax=axes[-1], fraction=0.046, pad=0.04)
    
    # Add arrows between plots
    fig.suptitle('Spectral Evolution Through FNO: Input → Lifting → Fourier Layers → Projection → Output', 
                 fontsize=13, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Spectral evolution visualization saved to {save_path}")


def create_riemann_sample(k1, k2, nx, nt, dx, dt):
    """Create a sample using Riemann initial condition."""
    riemann = Riemann(k1=k1, k2=k2)
    problem = Problem(nx=nx, nt=nt, dx=dx, dt=dt, ic=riemann, flow=Greenshield())
    solution = problem.solve(LaxHopf, batch_size=1, dtype=torch.float64, progressbar=False).cpu().numpy()
    
    # Convert to the format expected by the model
    input_grid = torch.from_numpy(solution[0]).to(torch.float32).unsqueeze(-1)  # (nt, nx, 1)
    target_grid = input_grid.clone()
    
    # Mask input like in GridDataset
    input_grid[1:, 1:-1] = -1  # mask all but the initial condition and boundary
    
    # Permute to (channels, nt, nx)
    sample_input = input_grid.permute(2, 0, 1)
    sample_target = target_grid.permute(2, 0, 1)
    
    return sample_input, sample_target


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize FNO spectral representations")
    parser.add_argument("--model_path", type=str, default="operator.pth", 
                        help="Path to trained model weights")
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--nt", type=int, default=250)
    parser.add_argument("--dx", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--model", type=str, default="FNO")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of different samples to visualize")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Base output directory")
    parser.add_argument("--riemann", action="store_true",
                        help="Use Riemann initial conditions instead of random")
    parser.add_argument("--riemann_configs", type=str, default=None,
                        help="Comma-separated k1:k2 pairs, e.g., '0.6:0.8,0.8:0.6'")
    return parser.parse_args()


def visualize_sample(model, sample_input, sample_target, output_folder):
    """Generate all visualizations for a single sample."""
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\n  Creating main spectral visualization...")
    plot_spectral_visualization(model, sample_input, sample_target, 
                                 f"{output_folder}/spectral_viz.png")
    
    print(f"  Creating detailed spectrum visualization...")
    plot_detailed_spectrum(model, sample_input, sample_target, 
                           f"{output_folder}/spectral_detailed.png")
    
    print(f"  Creating before/after comparison...")
    plot_before_after_spectrum(model, sample_input, sample_target, 
                                f"{output_folder}/spectral_before_after.png")
    
    print(f"  Creating spectral evolution visualization...")
    plot_spectral_evolution(model, sample_input, sample_target, 
                            f"{output_folder}/spectral_evolution.png")


def main():
    args = parse_args()
    
    # Create model
    print(f"Loading model from {args.model_path}...")
    model = create_model(args).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
        print("Model weights loaded successfully!")
    else:
        print(f"Warning: Model path {args.model_path} not found. Using random weights.")
    
    # Check if using Riemann initial conditions
    if args.riemann or args.riemann_configs:
        # Parse Riemann configurations
        if args.riemann_configs:
            configs = []
            for config in args.riemann_configs.split(','):
                k1, k2 = map(float, config.split(':'))
                configs.append((k1, k2))
        else:
            # Default Riemann configs if --riemann is specified without configs
            configs = [(0.6, 0.8), (0.8, 0.6)]
        
        print(f"\nGenerating visualizations for {len(configs)} Riemann initial conditions...")
        print("="*60)
        
        for idx, (k1, k2) in enumerate(configs):
            print(f"\n[Riemann {idx + 1}/{len(configs)}] k1={k1}, k2={k2}")
            
            # Create sample with Riemann IC
            sample_input, sample_target = create_riemann_sample(
                k1, k2, args.nx, args.nt, args.dx, args.dt
            )
            sample_input = sample_input.unsqueeze(0)  # Add batch dimension
            sample_target = sample_target.unsqueeze(0)
            
            # Create folder for this sample
            output_folder = f"{args.output_dir}/viz_riemann{idx + 1}"
            
            print(f"  → {output_folder}/")
            print(f"  Input shape: {sample_input.shape}")
            
            visualize_sample(model, sample_input, sample_target, output_folder)
        
        print("\n" + "="*60)
        print(f"All Riemann visualizations created!")
        print("="*60)
    
    else:
        # Original random sample mode
        print(f"Generating {args.num_samples} samples...")
        flux = Greenshields(vmax=1.0, rho_max=1.0)
        solver = Godunov(riemann_solver=LWRRiemannSolver(flux))
        
        dataset = GridDataset(solver, n_samples=args.num_samples, 
                              nx=args.nx, nt=args.nt, dx=args.dx, dt=args.dt)
        
        print(f"\nGenerating visualizations for {args.num_samples} different inputs...")
        print("="*60)
        
        # Generate visualizations for each sample
        for sample_idx in range(args.num_samples):
            sample_input, sample_target = dataset[sample_idx]
            sample_input = sample_input.unsqueeze(0)  # Add batch dimension
            sample_target = sample_target.unsqueeze(0)
            
            # Create folder for this sample
            output_folder = f"{args.output_dir}/viz{sample_idx + 1}"
            
            print(f"\n[Sample {sample_idx + 1}/{args.num_samples}] → {output_folder}/")
            print(f"  Input shape: {sample_input.shape}")
            
            visualize_sample(model, sample_input, sample_target, output_folder)
        
        print("\n" + "="*60)
        print(f"All visualizations created in {args.output_dir}/viz1 to {args.output_dir}/viz{args.num_samples}")
        print("="*60)


if __name__ == "__main__":
    main()

