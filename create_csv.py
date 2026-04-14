import numpy as np
import pandas as pd

# Load data
X = np.load("processed/x.npy")
y = np.load("processed/y.npy")

# NORMAL sample
for i in range(len(y)):
    if y[i] == 0:
        normal = X[i]
        break

pd.DataFrame(normal.reshape(-1)).to_csv("normal.csv", index=False)

# ARRHYTHMIA sample
for i in range(len(y)):
    if y[i] != 0:
        arr = X[i]
        break

pd.DataFrame(arr.reshape(-1)).to_csv("arrhythmia.csv", index=False)

print("🔥 Both test files ready")