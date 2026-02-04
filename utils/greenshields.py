import numpy as np
from scipy.optimize import curve_fit

class GreenshieldsProfiler:
    def __init__(self, data_path=None, data_array=None):
        """
        Args:
            data_path: Path to .npz file (optional)
            data_array: Numpy array (T, N, 3) (optional)
            Channels: 0:Flow, 1:Occupancy, 2:Speed (Note: PEMS usually has this order, check!)
        """
        if data_path:
            self.data = np.load(data_path)['data']
        elif data_array is not None:
            self.data = data_array
        else:
            raise ValueError("Provide data_path or data_array")
            
    def fit_physics(self):
        """
        Fits Greenshields Model: v = v_f * (1 - k/k_j)
        Convert to Flow-Density: q = k * v = k * v_f * (1 - k/k_j) = v_f * k - (v_f/k_j) * k^2
        
        Where:
            q: Flow (Channel 0)
            k: Density/Occupancy (Channel 1 approx)
            v: Speed (Channel 2)
            v_f: Free flow speed
            k_j: Jam density
            
        Returns:
            params: (v_f, k_j)
            r2: R-squared
        """
        # Flatten (T, N, 3) -> (T*N, 3)
        flat = self.data.reshape(-1, 3)
        
        # PEMS .npz usually: Flow, Occupancy, Speed? Or Flow, Speed, Occupancy?
        # Let's assume Flow, Occ, Speed based on common PEMS datasets.
        # But wait, our MockData was Flow, Speed, Occ.
        # Let's detect or enforce.
        # Standard PEMS04: 0:Flow, 1:Occupancy, 2:Speed.
        
        flow = flat[:, 0]
        # Occ is percentage, Density is veh/mile. They are proportional. 
        # We treat Occ as Density for this fitting.
        density = flat[:, 1] 
        speed = flat[:, 2]
        
        # Filter valid data (Speed > 0, Flow >= 0)
        mask = (speed > 0) & (flow >= 0)
        density_valid = density[mask]
        speed_valid = speed[mask]
        
        # Function: v = v_f * (1 - k/k_j)
        def greenshields_speed(k, v_f, k_j):
            return v_f * (1 - k/k_j)
            
        try:
            # Bounds: v_f > 0, k_j > 0
            popt, pcov = curve_fit(greenshields_speed, density_valid, speed_valid, bounds=(0, [200, 100]), p0=[60, 0.5])
            v_f, k_j = popt
            
            # Calc R2 for Speed
            pred_speed = greenshields_speed(density_valid, *popt)
            ss_res = np.sum((speed_valid - pred_speed) ** 2)
            ss_tot = np.sum((speed_valid - np.mean(speed_valid)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            print(f"[Greenshields] Fitted: v_f={v_f:.2f}, k_j={k_j:.4f} | R2={r2:.4f}")
            return v_f, k_j
            
        except Exception as e:
            print(f"[Greenshields] Fit Failed: {e}")
            return None, None

    def get_residual(self, flow, occ, speed, v_f, k_j):
        """
        Calculate physical residual based on fitted model.
        In perfect physics: Speed_obs approx Speed_model(Occ)
        But simpler consistency check: q = v * k.
        
        Refined Residual:
        Standard Greenshields implies a specific curve on q-k plane: q = v_f*k - (v_f/k_j)*k^2.
        
        However, the fundamental equation is q = k * v.
        Greenshields is a constitutive relation v = V(k).
        
        We can check TWO things:
        1. Fundamental Identity: Res_1 = Flow - (Speed * Occ * Coef_Unit_Conversion)
        2. Constitutive Relation: Res_2 = Speed - V_model(Occ)
        
        For Point 1 (Consistency), we might still need a linear coef if units mismatch (samples/5min vs mph).
        Let's perform a Robust Linear Fit for q = C * v * k on REAL data first to handle units.
        """
        pass
