# utils.py (Restored Version)
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # HPC 专用无头模式
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def concat_dict(acc, new_data):
    """Dictionary concatenation function."""
    def to_array(kk):
        if isinstance(kk, np.ndarray):
            return kk
        else:
            return np.asarray([kk])

    for k, v in new_data.items():
        if isinstance(v, dict):
            if k in acc:
                acc[k] = concat_dict(acc[k], v)
            else:
                acc[k] = concat_dict(dict(), v)
        else:
            v = to_array(v)
            if k in acc:
                acc[k] = np.concatenate([acc[k], v])
            else:
                acc[k] = np.copy(v)
    return acc

def get_scores_and_plot(scorer,
                        data_abs_xy,
                        activations,
                        directory,
                        filename,
                        plot_graphs=True,
                        nbins=20,
                        cm="jet",
                        sort_by_score_60=True):
    """
    Plotting function.
    Strictly follows the logic from the original DeepMind release:
    1. Calculate Rate Maps
    2. Calculate SAC (Spatial Autocorrelation) & Scores
    3. Sort units by Grid Score
    4. Plot Rate Map and SAC side-by-side
    """

    # Concatenate all trajectories
    # [Batch, Time, Features] -> [Total_Steps, Features]
    xy = data_abs_xy.reshape(-1, data_abs_xy.shape[-1])
    act = activations.reshape(-1, activations.shape[-1])
    n_units = act.shape[1]

    # 1. Get the rate-map for each unit
    print(f"Calculating Rate Maps for {n_units} units...", flush=True)
    s = [
        scorer.calculate_ratemap(xy[:, 0], xy[:, 1], act[:, i])
        for i in range(n_units)
    ]

    # 2. Get the scores and SAC
    print("Calculating Grid Scores and SAC...", flush=True)
    # scorer.get_scores returns: (score_60, score_90, max_60_mask, max_90_mask, sac)
    score_60, score_90, max_60_mask, max_90_mask, sac = zip(
        *[scorer.get_scores(rate_map) for rate_map in s])
    
    score_60 = np.array(score_60)
    score_90 = np.array(score_90)
    
    # 3. Sort by score if desired
    if sort_by_score_60:
        ordering = np.argsort(-score_60) # Descending order
    else:
        ordering = range(n_units)

    # 4. Plot
    # Original logic: plot all units. 
    # If n_units is large (e.g. 256), this creates a huge PDF.
    # We stick to original logic but you can limit 'n_units' loop if needed.
    cols = 16
    rows = int(np.ceil(n_units / cols))
    
    print(f"Plotting results to {filename}...", flush=True)
    fig = plt.figure(figsize=(24, rows * 4))
    
    for i in range(n_units):
        # Index for the unit to be plotted (based on sorting)
        index = ordering[i]
        
        # Subplot for Rate Map (Top or Odd rows in logic, here using separate indexing)
        # Original code used: rows * 2 because it plots SACs below RateMaps?
        # Actually original code:
        # rf = plt.subplot(rows * 2, cols, i + 1)
        # acr = plt.subplot(rows * 2, cols, n_units + i + 1)
        # This puts all RateMaps in the top half of the huge figure, and all SACs in the bottom half.
        
        rf = plt.subplot(rows * 2, cols, i + 1)
        acr = plt.subplot(rows * 2, cols, n_units + i + 1)
        
        title = "%d (%.2f)" % (index, score_60[index])
        
        # Plot the activation maps
        scorer.plot_ratemap(s[index], ax=rf, title=title, cmap=cm)
        
        # Plot the autocorrelation of the activation maps (SAC)
        scorer.plot_sac(
            sac[index],
            mask_params=max_60_mask[index],
            ax=acr,
            title=title,
            cmap=cm)

    # Save
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    full_path = os.path.join(directory, filename)
    with PdfPages(full_path, "w") as f:
        plt.savefig(f, format="pdf")
    plt.close(fig)
    
    return (score_60, score_90,
            np.array([np.mean(m) for m in max_60_mask]),
            np.array([np.mean(m) for m in max_90_mask]))

def plot_trajectory_comparison(ground_truth, predicted, save_path, epoch):
    """
    New function: Replicates Figure 1b (Trajectory Comparison).
    This was NOT in the original utils.py but is needed for your project.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Ground Truth (Light Blue, Thick)
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], 
            label='Ground Truth', color='#4ad', linewidth=3, alpha=0.7)
    
    # Predicted (Dark Blue, Thin)
    ax.plot(predicted[:, 0], predicted[:, 1], 
            label='Decoded Position', color='navy', linewidth=1.5, alpha=0.9)
    
    # Annotations
    ax.text(ground_truth[0, 0], ground_truth[0, 1], 'Start', fontsize=12, fontweight='bold')
    ax.text(ground_truth[-1, 0], ground_truth[-1, 1], 'End', fontsize=12, fontweight='bold')
    
    # Style
    limit = 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f"Trajectory Reconstruction (Epoch {epoch})")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Result] Trajectory plot saved to {save_path}")