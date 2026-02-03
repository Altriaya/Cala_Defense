import numpy as np
import torch

class AnomalyScorer:
    def __init__(self):
        self.stats = {} # Stores mean/std for each metric
        
    def fit(self, val_losses):
        """
        Calibrate on validation losses.
        Args:
            val_losses: Dict of list of losses, e.g. {'mae': [...], 'irm': [...]}
        """
        print("\n[Scorer] Calibrating on Validation Data...")
        for key, values in val_losses.items():
            v = np.array(values)
            mu = np.mean(v)
            sigma = np.std(v)
            if sigma < 1e-6: sigma = 1e-6
            
            self.stats[key] = {'mu': mu, 'sigma': sigma}
            print(f"  Metric '{key}': mu={mu:.4f}, sigma={sigma:.4f}")
            
    def score(self, losses):
        """
        Compute Z-Score for a single sample (or batch).
        Args:
            losses: Dict of losses, e.g. {'mae': 5.0, 'irm': 0.1}
        Returns:
            final_score (float)
            details (dict): individual z-scores
        """
        z_scores = []
        details = {}
        
        for key, val in losses.items():
            if key not in self.stats:
                continue
                
            mu = self.stats[key]['mu']
            sigma = self.stats[key]['sigma']
            
            z = (val - mu) / sigma
            z_scores.append(z)
            details[key] = z
            
        # Strategy: Max Z-Score
        # If ANY metric is violated significantly (>3 sigma), it's an anomaly.
        final_score = max(z_scores) if z_scores else 0.0
        return final_score, details

if __name__ == "__main__":
    scorer = AnomalyScorer()
    # Mock calibration
    val_data = {'mae': np.random.normal(10, 2, 100), 'irm': np.random.normal(0.5, 0.1, 100)}
    scorer.fit(val_data)
    
    # Test valid
    test_valid = {'mae': 11.0, 'irm': 0.55}
    s, d = scorer.score(test_valid)
    print(f"Valid Sample: Score={s:.2f}, Details={d}")
    
    # Test anomaly
    test_anom = {'mae': 20.0, 'irm': 0.5} # MAE huge
    s, d = scorer.score(test_anom)
    print(f"Anomaly Sample: Score={s:.2f}, Details={d}")
