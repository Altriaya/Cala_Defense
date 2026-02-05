import numpy as np

path = "d:/DLProjectHome/Cala_Defense/data/PEMS03/PEMS03.npz"
try:
    data = np.load(path)
    print(f"Keys: {list(data.keys())}")
    for k in data.keys():
        obj = data[k]
        if hasattr(obj, 'shape'):
            print(f"Key '{k}' shape: {obj.shape}")
        else:
            print(f"Key '{k}' type: {type(obj)}")
            
    # Check 'data' specifically
    if 'data' in data:
        print(f"Data sample (first element): {data['data'].flat[0]}")
        
except Exception as e:
    print(f"Error: {e}")
