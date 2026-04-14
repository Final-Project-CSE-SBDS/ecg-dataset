import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_dummy_data(filepath, num_samples=150, original_seq_len=250):
    """Generates a dummy ECG dataset for testing if a real one is not available."""
    logging.info(f"Generating dummy dataset at {filepath}...")
    np.random.seed(42)
    X = []
    y = []
    t = np.linspace(0, 10, original_seq_len)
    for _ in range(num_samples):
        is_normal = np.random.rand() > 0.4
        if is_normal:
            # Normal beat simulation (sine wave + noise)
            signal = np.sin(t * 2 * np.pi * 1.2) + 0.1 * np.random.randn(original_seq_len)
            label = 0 # Normal
        else:
            # Arrhythmia simulation (irregular waves + noise)
            signal = np.sin(t * 2 * np.pi * 1.5) + np.sin(t * 2 * np.pi * 3.0) + 0.5 * np.random.randn(original_seq_len)
            label = 1 # Arrhythmia
            
        X.append(signal)
        y.append(label)
        
    df = pd.DataFrame(X)
    df['Label'] = y
    
    # Introduce some missing values randomly for testing preprocessing
    df.iloc[5, 10] = np.nan
    df.iloc[12, 50] = np.nan
    df.iloc[20, 150] = np.nan
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logging.info("Dummy data generated successfully.")

def load_data(filepath):
    """Loads dataset from a CSV file."""
    if not os.path.exists(filepath):
        error_msg = f"File not found: '{filepath}'. Generating a dummy dataset for testing..."
        logging.warning(error_msg)
        generate_dummy_data(filepath)
        
    try:
        logging.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def handle_missing_values(df):
    """Handles missing values by interpolating or filling with 0."""
    logging.info("Checking for missing values in features...")
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logging.info(f"Found {missing_count} missing values. Fixing via forward-fill and back-fill...")
        # Fill missing values on rows (axis=1) since rows represent sequences over time
        df = df.ffill(axis=1).bfill(axis=1).fillna(0)
    else:
        logging.info("No missing values found.")
    return df

def segment_and_pad(features, window_size=200):
    """Segments or pads sequences to a fixed window size."""
    logging.info(f"Standardizing sequence lengths to {window_size} timesteps...")
    current_size = features.shape[1]
    
    if current_size == window_size:
        logging.info("Sequences already match the target window size.")
        return features
    elif current_size > window_size:
        # Truncate
        logging.info(f"Truncating sequences from {current_size} to {window_size}.")
        return features[:, :window_size]
    else:
        # Pad with zeros at the end
        logging.info(f"Padding sequences from {current_size} to {window_size} with zeros.")
        padding = np.zeros((features.shape[0], window_size - current_size))
        return np.hstack((features, padding))

def normalize_signals(features, method='minmax'):
    """Normalizes the signals on a per-sample basis."""
    logging.info(f"Normalizing signals using '{method}' scaling...")
    normalized = np.zeros_like(features)
    
    for i in range(features.shape[0]):
        sample = features[i, :].reshape(-1, 1)
        if method == 'minmax':
            scaler = MinMaxScaler()
        else: # z-score
            scaler = StandardScaler()
            
        normalized[i, :] = scaler.fit_transform(sample).flatten()
        
    return normalized

def plot_signals(raw_sample, processed_sample, title="ECG Signal: Before and After Preprocessing"):
    """Visualizes the signal before and after preprocessing."""
    try:
        plt.figure(figsize=(14, 5))
        
        # Raw Signal
        plt.subplot(1, 2, 1)
        plt.plot(raw_sample, color='blue', alpha=0.8)
        plt.title('Before Preprocessing (Raw)')
        plt.xlabel('Timesteps')
        plt.ylabel('Amplitude')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Processed Signal
        plt.subplot(1, 2, 2)
        plt.plot(processed_sample, color='green', alpha=0.8)
        plt.title('After Preprocessing (Normalized & Segmented)')
        plt.xlabel('Timesteps')
        plt.ylabel('Amplitude (0 to 1)')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = "processed/ecg_preprocessing_comparison.png"
        os.makedirs("processed", exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        logging.info(f"Visualization plot saved successfully to '{plot_path}'")
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting signals: {e}. Check if matplotlib is installed properly.")

def process_dataset(input_csv, target_col=-1, window_size=200):
    """Main function to pipeline the data processing steps."""
    try:
        # 1. Load Data
        df = load_data(input_csv)
        
        # Identify label column based on target_col index or name
        if isinstance(target_col, int):
            target_col_name = df.columns[target_col]
        else:
            target_col_name = target_col
            
        labels = df[target_col_name].values
        
        # 2. Encode Labels (Normal=0, Arrhythmia=1)
        logging.info("Encoding labels (Normal=0, Arrhythmia=1)...")
        if df[target_col_name].dtype == object:
            # Handle text labels, assuming 'normal' is class 0, others are 1
            encoded_labels = np.where(df[target_col_name].astype(str).str.lower().str.contains('normal'), 0, 1)
        else:
            # Handle binary/categorical numeric labels common in MIT-BIH (0 is Normal, >0 is something else)
            encoded_labels = (labels > 0).astype(int)
            
        features_df = df.drop(columns=[target_col_name])
        
        # 3. Handle Missing Values
        features_df = handle_missing_values(features_df)
        raw_features = features_df.values
        
        # Cache one sample for the final visualization
        sample_idx_to_plot = 0
        raw_sample = raw_features[sample_idx_to_plot].copy()
        
        # 4. Segment and Pad
        segmented_features = segment_and_pad(raw_features, window_size=window_size)
        
        # 5. Normalize Signals
        processed_features = normalize_signals(segmented_features, method='minmax')
        
        # 6. Plotting
        class_name = "Normal" if encoded_labels[sample_idx_to_plot] == 0 else "Arrhythmia"
        plot_signals(raw_sample, processed_features[sample_idx_to_plot], 
                     title=f"ECG Preprocessing Comparison (Class: {class_name})")
        
        return processed_features, encoded_labels

    except Exception as e:
        logging.error(f"Pipeline failed during preprocessing: {e}")
        raise

def save_and_split(features, labels, output_dir="processed", test_size=0.2):
    """Splits processed data into Train/Test sets and saves to CSV."""
    logging.info(f"Splitting dataset: Train ({(1-test_size)*100:.0f}%), Test ({test_size*100:.0f}%)...")
    
    # Stratified Train-Test Split ensures class proportions are equal in split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Reassemble to DataFrames for CSV saving
    train_df = pd.DataFrame(X_train)
    train_df['Label'] = y_train
    
    test_df = pd.DataFrame(X_test)
    test_df['Label'] = y_test
    
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logging.info(f"Data saved successfully to '{train_path}' and '{test_path}'")
    
    # Print the Final Statistics Output
    print("\n" + "=" * 50)
    print(" DATASET PROCESSING SUMMARY ")
    print("=" * 50)
    print(f"Total Original Samples: {features.shape[0]}")
    print(f"Shape of Processed Features: {features.shape}")
    print(f"Training Data Size: {X_train.shape[0]} samples")
    print(f"Testing Data Size:  {X_test.shape[0]} samples")
    
    normal_count = np.sum(labels == 0)
    arrhythmia_count = np.sum(labels == 1)
    print(f"Class Distribution:")
    print(f" - Normal (0):     {normal_count} ({normal_count/len(labels)*100:.1f}%)")
    print(f" - Arrhythmia (1): {arrhythmia_count} ({arrhythmia_count/len(labels)*100:.1f}%)")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    # Feel free to change these configuration variables
    INPUT_FILE = "data/raw_ecg_data.csv"
    WINDOW_SIZE = 200
    
    # Initialize the required data directory to avoid path errors
    os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)
    
    print("-> Starting ECG Dataset Processing Module\n")
    try:
        # Preprocess dataset pipeline
        features, labels = process_dataset(
            input_csv=INPUT_FILE, 
            target_col=-1, # Assuming the last column holds the label
            window_size=WINDOW_SIZE
        )
        
        # Split and export output
        save_and_split(
            features, 
            labels, 
            output_dir="processed", 
            test_size=0.2
        )
        
        print("Success! Dataset module has finished execution.")
        
    except Exception as e:
        print(f"\n[ERROR] Module execution aborted. Reason: {e}")
