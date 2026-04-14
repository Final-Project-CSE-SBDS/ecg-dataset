import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths
X_PATH = 'processed/x.npy'
Y_PATH = 'processed/y.npy'

def visualize_ecg():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("Data files not found. Please ensure processed/x.npy and processed/y.npy exist.")
        return

    # Load the processed arrays
    print("Loading prepared ECG data...")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    # Print shapes
    print("-" * 30)
    print(f"Features (X) shape: {X.shape}")
    print(f"Labels (y) shape: {y.shape}")
    
    # Display a few sample values
    print("-" * 30)
    print(f"Sample values from the segment at index 0 (first 10 numerical data points):")
    print(X[0, :10, 0])
    print(f"Actual Label for the first segment: Class {y[0]}")
    
    # Plot one ECG segment
    segment_idx = 0  
    
    plt.figure(figsize=(10, 4))
    
    # The X array is shape (num_samples, 200, 1), so we index into [segment_idx, :, 0]
    plt.plot(X[segment_idx, :, 0], color='#d62728', linewidth=1.5)
    
    plt.title(f"ECG Segment Index: {segment_idx} | Classified as: Class {y[segment_idx]}", fontsize=14, weight='bold')
    plt.xlabel("Sample Progression", fontsize=12)
    plt.ylabel("Normalized Amplitude", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot explicitly just in case standard display doesn't pop up
    plot_path = "sample_ecg_plot.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print("-" * 30)
    print(f"Successfully saved a plot of the ECG segment to '{plot_path}'")
    
    # Attempt to display the interactive plotting window
    plt.show()

if __name__ == '__main__':
    visualize_ecg()
