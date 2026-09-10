import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

def normalize_attribution(attr: np.ndarray) -> np.ndarray:
    """
    Normalizes attribution scores to [0, 1] range for visualization.
    Uses absolute values because both highly positive (pushes towards class)
    and highly negative (pushes away from class) attributions are important.
    """
    attr = np.abs(attr)
    max_val = attr.max()
    if max_val > 0:
        attr = attr / max_val
    return attr

def plot_1d_attribution(
    signal: np.ndarray, 
    attribution: np.ndarray, 
    title: str = "1D ECG Attribution", 
    save_path: str = None, 
    cmap_name: str = 'Reds',
    figsize: tuple = (15, 8)
):
    """
    Plots the 12-lead ECG signal with attribution overlay.
    
    Args:
        signal: 1D signal of shape (12, signal_length)
        attribution: Attribution mask of shape (12, signal_length)
        title: Title of the plot
        save_path: Optional path to save the figure
        cmap_name: Matplotlib colormap for attribution
        figsize: Figure size
    """
    num_leads, signal_length = signal.shape
    fig, axes = plt.subplots(num_leads, 1, figsize=figsize, sharex=True)
    if num_leads == 1:
        axes = [axes]
        
    x = np.arange(signal_length)
    cmap = plt.get_cmap(cmap_name)
    
    # Normalize attribution per lead or globally? Globally is better to compare leads.
    norm_attr = normalize_attribution(attribution)
    
    for i in range(num_leads):
        ax = axes[i]
        y = signal[i]
        attr_y = norm_attr[i]
        
        # Create line segments for color mapping
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Map attribution to color
        # We use the average attribution of the segment's two endpoints
        segment_attr = (attr_y[:-1] + attr_y[1:]) / 2.0
        
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(segment_attr)
        lc.set_linewidth(2)
        
        ax.add_collection(lc)
        ax.set_xlim(0, signal_length)
        ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
        ax.set_ylabel(f'Lead {i+1}')
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time steps")
    fig.suptitle(title)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_2d_attribution(
    image: np.ndarray, 
    attribution: np.ndarray, 
    title: str = "2D Attribution", 
    save_path: str = None, 
    cmap_name: str = 'jet',
    alpha: float = 0.5,
    figsize: tuple = (15, 10)
):
    """
    Plots the 2D feature map (e.g. Spectrogram/GAF) with attribution overlay.
    
    Args:
        image: Original 2D input (leads, H, W)
        attribution: 2D attribution mask (leads, H, W)
    """
    num_leads = image.shape[0]
    cols = 3
    rows = (num_leads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    norm_attr = normalize_attribution(attribution)
    
    for i in range(num_leads):
        ax = axes[i]
        
        # Base image
        img = image[i]
        ax.imshow(img, cmap='gray', aspect='auto')
        
        # Overlay
        if norm_attr.ndim == 2:
            attr_img = norm_attr
        elif norm_attr.shape[0] == 1:
            attr_img = norm_attr[0]
        else:
            attr_img = norm_attr[i]
            
        # We mask low attributions to make the original image visible
        overlay = ax.imshow(attr_img, cmap=cmap_name, aspect='auto', alpha=alpha * attr_img)
        
        ax.set_title(f"Lead {i+1}")
        ax.axis('off')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(title)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_translated_attribution(
    signal: np.ndarray,
    regions: dict,
    title: str = "Lead-Specific ECG Grad-CAM Attribution Mapping",
    save_path: str = None,
    figsize: tuple = (15, 14)
):
    """
    Plots the 12-lead ECG signal and highlights the lead-specific high-attribution regions.
    """
    num_leads = signal.shape[0]
    fig, axes = plt.subplots(num_leads, 1, figsize=figsize, sharex=True)
    
    if num_leads == 1:
        axes = [axes]
        
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    if num_leads != 12:
        lead_names = [f"Lead {i+1}" for i in range(num_leads)]
        
    # Find max attribution strength for normalization
    max_strength = max([r.get('attribution_strength', 1.0) for r in regions.values()]) if regions else 1.0
    if max_strength == 0:
        max_strength = 1.0
        
    # Styling options
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    
    for i in range(num_leads):
        ax = axes[i]
        # Plot signal with a clean dark slate color
        ax.plot(signal[i], color='#2B2D42', linewidth=1.2, alpha=0.9)
        
        region = regions.get(i)
        if region:
            start_s = region['start_sample']
            end_s = region['end_sample']
            waves = ", ".join(region['overlapping_waves'])
            strength = region['attribution_strength']
            
            # Normalize alpha based on attribution strength
            alpha_val = 0.15 + 0.35 * (strength / max_strength)
            
            # Highlight region with warm crimson / coral tone
            ax.axvspan(start_s, end_s, color='#EF233C', alpha=alpha_val, label='Attribution')
            
            # Draw dashed boundaries for clean look
            ax.axvline(start_s, color='#EF233C', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axvline(end_s, color='#EF233C', linestyle='--', linewidth=0.8, alpha=0.7)
            
            # Label component and strength
            ylim = ax.get_ylim()
            y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.15
            ax.text(
                (start_s + end_s) / 2, 
                y_pos, 
                f"{waves}\n(Attr: {strength:.2f})", 
                color='#D90429', 
                fontsize=8, 
                fontweight='bold',
                horizontalalignment='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
            )
                    
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=25, va='center', fontweight='bold', fontsize=10, color='#2B2D42')
        ax.grid(True, color='#E0E0E0', linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Remove top and right splines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    axes[-1].set_xlabel("Time steps (Samples)", fontweight='bold', fontsize=10, color='#2B2D42')
    fig.suptitle(title, fontsize=14, fontweight='bold', color='#1A1A1D', y=0.98)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
