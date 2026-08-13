import numpy as np
import pandas as pd

def inspect_npy_files():
    data_dir = "processed_76_channels"
    
    print(f"Loading data from {data_dir}...\n")
    
    # Load the arrays
    test_X = np.load(f"{data_dir}/test_X.npy")
    test_y = np.load(f"{data_dir}/test_y.npy")
    
    print(f"Full Array Shapes:")
    print(f"  test_X: {test_X.shape}")
    print(f"  test_y: {test_y.shape}\n")
    
    # Wrap the first 5 rows of the feature matrix in a DataFrame
    print("=== test_X.npy (First 5 Rows / 106 Features) ===")
    
    # We will just print the first 10 columns so it fits nicely on your screen
    df_head = pd.DataFrame(test_X[:5, :10]) 
    
    # Rename columns just to make it clear what we are looking at
    df_head.columns = [f"Feature_{i}" for i in range(10)]
    print(df_head.to_string())
    print("... (and 96 more columns) ...\n")
    
    # Print the first 15 labels
    print("=== test_y.npy (First 15 Labels) ===")
    print(test_y[:15])
    print(f"\nTotal Anomalies in this whole file: {test_y.sum():,} out of {len(test_y):,}")

if __name__ == "__main__":
    inspect_npy_files()
