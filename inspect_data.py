import numpy as np

try:
    data = np.load('d:/DLProjectHome/Cala_Defense/data/PEMS04/PEMS04.npz')
    print("Keys:", data.files)
    for k in data.files:
        print(f"Key '{k}': Shape {data[k].shape}")
except Exception as e:
    print(f"Error: {e}")
