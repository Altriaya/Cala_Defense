from utils.greenshields import GreenshieldsProfiler
import numpy as np

try:
    path = 'd:/DLProjectHome/Cala_Defense/data/PEMS04/PEMS04.npz'
    profiler = GreenshieldsProfiler(data_path=path)
    # 1. Fit Greenshields (v-k relation)
    v_f, k_j = profiler.fit_physics()
    
    # 2. Check Fundamental Diagram (q-vk relation)
    # Load data manually to check linear coef
    data = np.load(path)['data'] # (T, N, 3) 0:Flow, 1:Occ, 2:Speed
    # PEMS04 is usually Cost, Flow, Speed? Or Flow, Occ, Speed?
    # Actually PEMS04.npz provided by ASTGCN/etc usually has 3 channels: Flow, Occ, Speed.
    
    # Let's verify correlation
    flat = data.reshape(-1, 3)
    flow = flat[:, 0]
    occ = flat[:, 1]
    speed = flat[:, 2]
    
    # q ~ v * k
    vk = speed * occ
    # Linear Fit
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression(fit_intercept=False) # Zero intercept physically
    lr.fit(vk.reshape(-1, 1), flow)
    print(f"[Fundamental] Flow = {lr.coef_[0]:.4f} * (Speed * Occ)")
    print(f"[Fundamental] Score: {lr.score(vk.reshape(-1, 1), flow):.4f}")

except Exception as e:
    print(e)
