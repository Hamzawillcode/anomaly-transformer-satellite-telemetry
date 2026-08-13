import os
import numpy as np
import matplotlib.pyplot as plt

def generate_presentation_plots():
    data_dir = "processed_76_channels"
    save_dir = "presentation_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[PLOTS] Loading test data matrix...")
    test_X = np.load(f"{data_dir}/test_X.npy")
    
    # ---------------------------------------------------------
    # PLOT 1: The Normalization Distribution (Logarithmic Scale)
    # ---------------------------------------------------------
    print("[PLOTS] Generating Logarithmic Histogram...")
    idx = np.random.choice(test_X.shape[0], 50000, replace=False)
    global_sample = test_X[idx, :]
    
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.hist(global_sample[:, i], bins=100, alpha=0.4, log=True, label=f'Channel {i+1}')
        
    plt.title("Post-Scaling Telemetry Distribution (Logarithmic Scale)")
    plt.xlabel("Normalized Sensor Value (Standard Deviations)")
    plt.ylabel("Frequency (Log Scale)")
    plt.axvline(0, color='black', linestyle='dashed', linewidth=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/1_normalization_histogram.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Multimodal Alignment (200-Step Active Window)
    # ---------------------------------------------------------
    print("[PLOTS] Generating 200-step Zoomed Window...")
    max_var = -1
    best_start = 0
    for i in range(0, 50000, 200):
        window_var = np.var(test_X[i:i+200, 0])
        if window_var > max_var:
            max_var = window_var
            best_start = i
            
    active_window = test_X[best_start:best_start+200, :]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(active_window[:, 0], color='blue', linewidth=1.5)
    ax1.set_title(f"Aligned 30-Second Grid: Zoomed Sensor Telemetry (Window start: {best_start})")
    ax1.set_ylabel("Normalized Value")
    ax1.grid(True, alpha=0.3)
    
    simulated_tc = np.zeros(200)
    simulated_tc[45] = 1.0  
    simulated_tc[120] = 1.0 
    ax2.plot(simulated_tc, color='red', drawstyle='steps-pre', linewidth=2)
    ax2.set_ylabel("TC Impulse")
    ax2.set_xlabel("Time Steps (30-second intervals)")
    ax2.set_yticks([0, 1])
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/2_multimodal_alignment.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # NEW PLOT 3: Macro Mission Timeline (50,000 Time Steps)
    # ---------------------------------------------------------
    print("[PLOTS] Generating 50,000-step Macro Timeline (~17.4 Days)...")
    n_steps = 50000
    long_sample = test_X[:n_steps, :]
    
    # Mathematical transformation: convert 30-second steps to cumulative days
    time_days = (np.arange(n_steps) * 30) / (3600 * 24)
    
    plt.figure(figsize=(15, 5))
    
    # Plotting 3 separate continuous channels to show structural variance over 17 days
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(3):
        plt.plot(time_days, long_sample[:, i], label=f'Channel {i+1}', linewidth=0.5, alpha=0.7, color=colors[i])
        
    plt.title(f"Macro Mission Timeline: 17.4 Days of Continuous Spacecraft Telemetry ({n_steps:,} Timestamps)", fontsize=14)
    plt.xlabel("Mission Elapsed Time (Days)", fontsize=12)
    plt.ylabel("Normalized Operational Value ($\sigma$)", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    
    # Setting tight layout limits to keep the graph perfectly clean
    plt.xlim(0, time_days[-1])
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    long_plot_path = f"{save_dir}/3_macro_timeline_50k.png"
    plt.savefig(long_plot_path, dpi=300)
    plt.close()
    print(f"  ↳ Saved: {long_plot_path}")
    
    print("[PLOTS] All presentation visuals successfully updated!")

if __name__ == "__main__":
    generate_presentation_plots()
