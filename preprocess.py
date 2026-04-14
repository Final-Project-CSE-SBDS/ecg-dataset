import os
import wfdb
import numpy as np
from collections import Counter

# Define configuration
DATA_DIR = 'data/'
PROCESSED_DIR = 'processed/'
WINDOW_SIZE = 200

# Convert annotation symbols into numeric labels using a mapping (AAMI standard)
# N: Normal, S: Supraventricular, V: Ventricular, F: Fusion, Q: Unknown
SYMBOL_TO_LABEL = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,         # Supraventricular
    'V': 2, 'E': 2,                         # Ventricular
    'F': 3,                                 # Fusion
    '/': 4, 'f': 4, 'Q': 4,                 # Unknown
}

def load_and_preprocess_records(records):
    X_all = []
    y_all = []

    for record_name in records:
        record_path = os.path.join(DATA_DIR, record_name)
        
        # 1. Load multiple ECG records from a local data folder
        if not os.path.exists(record_path + '.hea'):
            print(f"Warning: Record {record_name} not found in {DATA_DIR}. Downloading automatically from PhysioNet...")
            try:
                wfdb.dl_database('mitdb', dl_dir=DATA_DIR, records=[record_name])
            except Exception as e:
                print(f"Failed to download record {record_name}: {e}")
                continue
        
        # 2. Read ECG signals and annotations
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
        except Exception as e:
            print(f"Error reading record {record_name}: {e}")
            continue

        # Extract MLII channel (usually the first channel)
        signal = record.p_signal[:, 0]
        
        # 3. Normalize the ECG signal
        signal = (signal - np.mean(signal)) / np.std(signal)

        symbols = annotation.symbol
        samples = annotation.sample

        # 4. Segment the signal into fixed window sizes of 200 samples
        # Here we center the window around the beat annotation
        half_window = WINDOW_SIZE // 2
        
        for i, sample in enumerate(samples):
            sym = symbols[i]
            
            # Skip unmapped symbols
            if sym not in SYMBOL_TO_LABEL:
                continue
                
            start = sample - half_window
            end = sample + half_window
            
            # Ensure window is within signal bounds
            if start < 0 or end > len(signal):
                continue
                
            segment = signal[start:end]
            
            if len(segment) != WINDOW_SIZE:
                continue
                
            # 5 & 6. Assign label based on annotation symbols and convert using mapping
            label = SYMBOL_TO_LABEL[sym]
            
            X_all.append(segment)
            y_all.append(label)

    # 7. Combine all segments from all records into a single dataset
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    return X_all, y_all

if __name__ == '__main__':
    # Define records to process (100 to 110 inclusive)
    records_to_load = [str(i) for i in range(100, 111)]
    
    print(f"Loading and processing records: {records_to_load}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    X, y = load_and_preprocess_records(records_to_load)
    
    if len(X) == 0:
        print(f"\nNo data extracted. Ensure the MIT-BIH files (.dat, .hea, .atr) for the records {records_to_load} are inside the 'data' folder.")
    else:
        # 8. Reshape input into (num_samples, 200, 1)
        X = X.reshape(-1, WINDOW_SIZE, 1)
        
        # 9. Save processed data as processed/x.npy and processed/y.npy
        x_path = os.path.join(PROCESSED_DIR, 'x.npy')
        y_path = os.path.join(PROCESSED_DIR, 'y.npy')
        
        np.save(x_path, X)
        np.save(y_path, y)
        
        print("\nProcessing complete!")
        print(f"Saved {x_path}")
        print(f"Saved {y_path}")
        print("-" * 30)
        
        # 10. Print dataset shape and class distribution
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print("\nClass distribution:")
        
        class_counts = Counter(y)
        total_samples = len(y)
        for cls, count in sorted(class_counts.items()):
            print(f"Class {cls}: {count} samples ({count/total_samples*100:.2f}%)")
