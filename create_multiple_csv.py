import numpy as np
import pandas as pd

# Load dataset
X = np.load("processed/x.npy")
y = np.load("processed/y.npy")

# Create multiple arrhythmia samples
count = 0
for i in range(len(y)):
    if y[i] != 0:
        df = pd.DataFrame(X[i].reshape(-1))
        df.to_csv(f"arr_{count}.csv", index=False)
        count += 1

    if count == 5:
        break

print("🔥 5 arrhythmia CSV files created")