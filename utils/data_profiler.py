import numpy as np
from sklearn.linear_model import LinearRegression

class PEMSDataProfiler:
    def __init__(self, data):
        """
        Args:
            data: numpy array of shape (B, T, N, 3) or (N_samples, 3)
                  Channels expected: 0:Flow, 1:Speed, 2:Occupancy
        """
        self.data = data
    
    def analyze_physics(self):
        """
        Fits q = C * (v * rho) to find C.
        Returns:
            coef (float): The coefficient C.
            r2 (float): R-squared validaton of the physics model.
        """
        # Flatten data to (N_total, 3)
        if self.data.ndim > 2:
            flat_data = self.data.reshape(-1, 3)
        else:
            flat_data = self.data
            
        flow = flat_data[:, 0]
        speed = flat_data[:, 1]
        occ = flat_data[:, 2]
        
        # X = v * rho
        physics_term = speed * occ
        
        # Fit Linear Regression: Flow = C * physics_term + Bias (Bias should be near 0)
        model = LinearRegression(fit_intercept=True)
        model.fit(physics_term.reshape(-1, 1), flow)
        
        coef = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(physics_term.reshape(-1, 1), flow)
        
        print(f"[Profiler] Physics Model: Flow = {coef:.4f} * (Speed * Occ) + {intercept:.4f}")
        print(f"[Profiler] R2 Score: {r2:.4f}")
        
        return coef, intercept

if __name__ == "__main__":
    from mock_data import MockPEMSDataset
    dataset = MockPEMSDataset(num_samples=100)
    profiler = PEMSDataProfiler(dataset.get_all_data())
    profiler.analyze_physics()
