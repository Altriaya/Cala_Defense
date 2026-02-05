import torch
import numpy as np
import sys
import os
import traceback

# Path setup
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)

# Utils
from utils.greenshields import GreenshieldsProfiler
from utils.physical_augmenter import PhysicalAugmenter
from utils.environment_splitter import TimeBasedSplitter
from utils.data_profiler import PEMSDataProfiler
from utils.spectral_physics import SpectralProfiler, SpectralAugmenter

# Models & Trainers
from consistency.mae_model import PhysicsMAE
from consistency.train_mae import MAETrainer
from invariance.graph_learner import InvariantGraphLearner
from invariance.train_irm import IRMTrainer
from models import TimesNetForecaster
from attack_adapter import BackTimeInjector
from detection.scorer import AnomalyScorer
from detection.baselines import PCADetector, IForestDetector

from sklearn.metrics import roc_auc_score, roc_curve

class DefensePipeline:
    def __init__(self, dataset="PEMS04"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Initializing on {self.device}...")
        
        # State
        self.dataset = dataset
        self.profiler = None
        self.mae = None
        self.irm = None
        self.scorer = AnomalyScorer()
        
        # Augmenter
        self.augmenter = None
        self.in_dim = 1
        
    def prepare_data(self):
        print(f"\n[Phase 0] Preparing Real {self.dataset} Data...")
        if self.dataset == "PEMS04":
            data_path = "d:/DLProjectHome/Cala_Defense/data/PEMS04/PEMS04.npz"
        elif self.dataset == "PEMS08":
             data_path = "d:/DLProjectHome/Cala_Defense/data/PEMS08/PEMS08.npz"
        elif self.dataset == "PEMS03":
             data_path = "d:/DLProjectHome/Cala_Defense/data/PEMS03/PEMS03.npz"
        else:
             raise ValueError(f"Unknown dataset: {self.dataset}")

        # Load Real Data
        try:
            # 1. Load PEMS Data
            data_raw = np.load(data_path)['data'] 
            # Slice for demo speed
            demo_steps = 4000 
            
            # Check Channels
            if data_raw.ndim == 3:
                channels = data_raw.shape[2]
            else: 
                channels = 1 
                
            print(f"  Data loaded: {data_raw.shape} (Channels: {channels})")
            
            # Universal Spectral Physics (Only for 1-Channel Fallback)
            # Experiment showed adding Saliency to 3-Channel data adds noise and degrades AUC.
            # So we only use it when we lack Domain Physics.
            
            # Slice actual data
            data = data_raw[:demo_steps, :10] # (T, N, C)
            
            if channels >= 3:
                # 2. Universal Spectral Physics (Replacing Domain Physics)
                # Testing hypothesis: Spectral Consistency > Greenshields Physics for this attack.
                print("  [Physics] Fitting Universal Spectral Physics (Saliency)...")
                
                # Check 1-channel logic availability
                if not hasattr(self, 'profiler_spectral'):
                    self.profiler_spectral = SpectralProfiler()
                
                # Channel 0 is always Flow
                flow_data = data[..., 0] # (T, N)
                self.profiler_spectral.fit(flow_data)
                saliency = self.profiler_spectral.get_residual(flow_data) # (T, N)
                
                # Stack as Channel 4
                saliency_exp = saliency[..., np.newaxis]
                aug_data = np.concatenate([data, saliency_exp], axis=-1) # (T, N, 4)
                
                print(f"  [Augmenter] Replaced Greenshields with Spectral Saliency. Shape: {aug_data.shape}")
                
                # Use Spectral Augmenter
                self.augmenter = SpectralAugmenter(self.profiler_spectral)
                self.in_dim = 4
                
                # Disable Greenshields explicitly
                self.profiler = None
                
            else:
                # 1 Channel (Flow Only) -> Spectral Only
                print("  [Physics] Domain Physics N/A. Using Spectral Head Only.")
                if not hasattr(self, 'profiler_spectral'):
                    self.profiler_spectral = SpectralProfiler()
                
                if data.ndim == 2: data = data[..., np.newaxis]
                flow_data = data[..., 0] # (T, N)
                
                self.profiler_spectral.fit(flow_data)
                saliency = self.profiler_spectral.get_residual(flow_data) # (T, N)
                
                # Stack
                aug_data = np.stack([flow_data, saliency], axis=-1) # (T, N, 2)
                
                self.augmenter = SpectralAugmenter(self.profiler_spectral)
                self.in_dim = 2
            
            # 4. Normalize
            self.data_mean = np.mean(aug_data, axis=(0, 1), keepdims=True)
            self.data_std = np.std(aug_data, axis=(0, 1), keepdims=True)
            self.data_std[self.data_std < 1e-6] = 1.0
            
            # Expand mean/std for broadcasting
            self.data_mean = self.data_mean[np.newaxis, ...]
            self.data_std = self.data_std[np.newaxis, ...]

            if channels == 1:
                 norm_data = (aug_data - self.data_mean[0,0]) / self.data_std[0,0]
            else:
                 norm_data = (aug_data - self.data_mean[0,0]) / self.data_std[0,0]
            
            # Save raw slice
            data_slice = norm_data[:100]
            
            # 4. Normalize
            self.data_mean = np.mean(aug_data, axis=(0, 1), keepdims=True)
            self.data_std = np.std(aug_data, axis=(0, 1), keepdims=True)
            self.data_std[self.data_std < 1e-6] = 1.0
            
            # Expand mean/std for broadcasting
            self.data_mean = self.data_mean[np.newaxis, ...]
            self.data_std = self.data_std[np.newaxis, ...]

            if channels == 1: # Now 2 channels
                norm_data = (aug_data - self.data_mean[0,0]) / self.data_std[0,0]
            else:
                norm_data = (aug_data - self.data_mean[0,0]) / self.data_std[0,0]
            
            # Save raw slice
            data_slice = norm_data[:100]
            
            # 5. Windowing (Sliding Window)
            seq_len = 24
            samples = []
            # Create samples (Clean)
            # Use Stride=1 for Maximum Data (Solve IRM Starvation)
            print("  Windowing with Stride=1...")
            for t in range(0, norm_data.shape[0] - seq_len, 1): 
                samples.append(norm_data[t:t+seq_len])
            
            samples = np.array(samples) # (NumSamples, 24, N, 4)
            print(f"  Windowed Samples: {samples.shape}")
            
            return samples, data_slice # Return raw for splitter if needed
            
        except Exception as e:
            print(f"[Error] Data Prep Failed: {e}")
            traceback.print_exc() # Added
            sys.exit(1)

    def train_defense(self, train_data, raw_sequence_slice):
        print("\n[Phase 1] Training Defense Models (Scientific)...")
        
        # 1. Train MAE
        print(f"  [1.1] Training Physics-MAE (in_c={self.in_dim})...")
        self.mae = PhysicsMAE(seq_len=24, in_channels=self.in_dim, patch_size=1, mask_ratio=0.4).to(self.device)
        mae_trainer = MAETrainer(self.mae, self.device)
        
        B, T, N, C = train_data.shape
        mae_input = train_data.transpose(0, 2, 1, 3).reshape(B*N, T, C)
        
        # Larger batch for dense data
        mae_dataset = torch.utils.data.TensorDataset(torch.from_numpy(mae_input).float())
        mae_loader = torch.utils.data.DataLoader(
            mae_dataset, batch_size=256, shuffle=True
        )
        mae_trainer.train_model(mae_loader, epochs=5) 
        
        # 2. Train IRM
        print(f"  [1.2] Training Invariant Graph Learner (IRM) (in_c={self.in_dim})...")
        
        splitter = TimeBasedSplitter(steps_per_day=288)
        
        envs = [[], [], []] # AM, PM, Off
        for i in range(len(train_data)):
            t_start = i * 1 # Stride 1
            env_id = splitter.get_env_id(t_start)
            envs[env_id].append(train_data[i])
            
        env_tensors = [torch.from_numpy(np.array(e)).float().to(self.device) for e in envs if len(e) > 0]
        print(f"    IRM Environments: {[e.shape[0] for e in env_tensors]}")
        
        self.irm = InvariantGraphLearner(num_nodes=N, in_channels=self.in_dim).to(self.device)
        optimizer = torch.optim.Adam(self.irm.parameters(), lr=0.005)
        self.irm.train()
        
        for ep in range(10):
            total_loss = 0
            min_len = min([len(e) for e in env_tensors])
            batch_size = 64 # Larger batch for stable gradient
            
            iter_limit = min(50, min_len // batch_size) 

            for _ in range(iter_limit):
                env_losses = []
                for env_data in env_tensors:
                    idx = torch.randperm(len(env_data))[:batch_size]
                    batch = env_data[idx] 
                    pred, _ = self.irm(batch)
                    loss = torch.mean((pred - batch)**2)
                    env_losses.append(loss)
                
                loss_stack = torch.stack(env_losses)
                mean_loss = torch.mean(loss_stack)
                var_loss = torch.var(loss_stack)
                loss = mean_loss + 100.0 * var_loss # Stronger Penalty
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if ep % 2 == 0:
                print(f"    [IRM Epoch {ep}] Loss: {total_loss:.4f}")

        # 3. Train Spectral Filter (PCA) - New Module!
        print("  [1.3] Fitting Spectral Filter (PCA)...")
        from detection.baselines import PCADetector
        self.pca = PCADetector(n_components=0.95)
        self.pca.fit(train_data)

    def calibrate(self, val_data):
        print("\n[Phase 2] Calibrating Scorer (Val Data)...")
        self.mae.eval()
        self.irm.eval()
        losses = {'mae': [], 'irm': [], 'pca': []}
        
        B, T, N, C = val_data.shape
        
        # MAE (Chunked)
        mae_input = val_data.transpose(0, 2, 1, 3).reshape(B*N, T, C)
        t_mae = torch.from_numpy(mae_input).float().to(self.device)
        with torch.no_grad():
            batches = torch.split(t_mae, 1024)
            for b in batches:
                 p, m, _ = self.mae(b)
                 l = torch.mean((p-b)**2, dim=(1,2))
                 losses['mae'].extend(l.cpu().numpy())
        
        # IRM
        t_irm = torch.from_numpy(val_data).float().to(self.device)
        with torch.no_grad():
             # Split IRM too if huge
             batches = torch.split(t_irm, 128)
             for b in batches:
                 p, _ = self.irm(b)
                 l = torch.mean((p-b)**2, dim=(1,2,3))
                 losses['irm'].extend(l.cpu().numpy())
        
        # PCA
        # PCA scores are numpy already
        l_pca = self.pca.score(val_data)
        losses['pca'].extend(l_pca)
             
        self.scorer.fit(losses)

    def detect_score(self, x_norm):
        self.mae.eval()
        self.irm.eval()
        with torch.no_grad():
            B, T, N, C = x_norm.shape
            # MAE
            x_mae = x_norm.transpose(0, 2, 1, 3).reshape(B*N, T, C)
            t_mae = torch.from_numpy(x_mae).float().to(self.device)
            p_m, m, _ = self.mae(t_mae)
            loss_mae = torch.mean((p_m - t_mae)**2, dim=(1,2)).reshape(B, N).mean(dim=1).cpu().numpy()
            
            # IRM
            t_irm = torch.from_numpy(x_norm).float().to(self.device)
            p_i, _ = self.irm(t_irm)
            loss_irm = torch.mean((p_i - t_irm)**2, dim=(1,2,3)).cpu().numpy()
            
            # PCA
            loss_pca = self.pca.score(x_norm)
            
            # Tri-Shield: Logic OR (Max) or Logic AND (Avg)?
            # PCA is great for "Smooth" triggers.
            # MAE is great for "Spiky" triggers.
            # IRM is great for "Context" shifts.
            
            z_mae = (loss_mae - self.scorer.stats['mae']['mu']) / self.scorer.stats['mae']['sigma']
            z_irm = (loss_irm - self.scorer.stats['irm']['mu']) / self.scorer.stats['irm']['sigma']
            z_pca = (loss_pca - self.scorer.stats['pca']['mu']) / self.scorer.stats['pca']['sigma']
            
            # Weighted Mean Fusion (Robustness > Sensitivity)
            # Max Fusion was too sensitive to noise in Saliency channel.
            
            scores = (z_mae + z_irm + z_pca) / 3.0
            
            return scores
            # MAE is great for "Noise" triggers.
            # IRM is great for "Distribution Shift".
            # Max increases FPR (Union of errors). Mean suppresses uncorrelated noise (Intersection-ish).
            # Switch to Mean for stability.
            # Z-Score Fusion
            stats = self.scorer.stats
            z_mae = (loss_mae - stats['mae']['mu']) / stats['mae']['sigma']
            z_irm = (loss_irm - stats['irm']['mu']) / stats['irm']['sigma']
            z_pca = (loss_pca - stats['pca']['mu']) / stats['pca']['sigma']

            scores = np.mean([z_mae, z_irm, z_pca], axis=0)
            return scores

    def train_baselines(self, train_data):
        print("\n[Phase 3.5] Training Baselines (PCA & IForest)...")
        pca = PCADetector()
        pca.fit(train_data)
        iforest = IForestDetector()
        iforest.fit(train_data)
        print("  Baselines Trained.")
        return {'pca': pca, 'iforest': iforest}

    def compute_metrics(self, pred, true):
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)
        
        # WMAPE (Weighted MAPE) to avoid denominator trap
        # sum(|pred - true|) / sum(|true|)
        sum_abs_diff = np.sum(np.abs(pred - true))
        sum_abs_true = np.sum(np.abs(true))
        
        if sum_abs_true > 1e-6:
            wmape = (sum_abs_diff / sum_abs_true) * 100.0
        else:
            wmape = 0.0
            
        return rmse, wmape

    def run_eval(self):
        # 1. Prep
        data, raw_slice = self.prepare_data()
        
        n = len(data)
        train_d = data[:int(0.5*n)]
        val_d = data[int(0.5*n):int(0.75*n)]
        test_d = data[int(0.75*n):]
        
        # 2. Train Defense
        self.train_defense(train_d, raw_slice)
        
        # 3. Calibrate
        self.calibrate(val_d)
        
        # 3.5 Train Baselines
        baselines = self.train_baselines(train_d)
        
        # 4. Train Victim
        print("\n[Phase 3] Training Victim Model (TimesNet)...")
        victim = TimesNetForecaster(in_dim=1, out_dim=12).to(self.device)
        optimizer = torch.optim.Adam(victim.parameters(), lr=0.01)
        v_train_in = torch.from_numpy(train_d[:500, :24, 0, 0:1]).float().to(self.device)
        victim.train()
        for e in range(30):
             p = victim(v_train_in[:,:12])
             l = torch.nn.functional.mse_loss(p, v_train_in[:,12:,0])
             optimizer.zero_grad(); l.backward(); optimizer.step()
        print("  Victim Trained.")
        
        # 5. Attack & Lambda Sweep
        print("\n[Phase 4] Scientific Evaluation (Pareto & Baselines)...")
        
        atk_subset_indices = np.random.choice(len(train_d), size=min(len(train_d), 500), replace=False)
        atk_subset = train_d[atk_subset_indices]
        atk_loader = torch.utils.data.DataLoader(torch.from_numpy(atk_subset).float().to(self.device), batch_size=32, shuffle=True)
        
        t_test = torch.from_numpy(test_d).float().to(self.device)
        
        # Latency Check (Clean)
        import time
        t0 = time.time()
        scores_clean = self.detect_score(test_d)
        dt = time.time() - t0
        latency_ms = (dt / len(test_d)) * 1000
        print(f"\n[Efficiency] Inference Latency: {latency_ms:.2f} ms/sample")
        
        pca_clean = baselines['pca'].score(test_d)
        if_clean = baselines['iforest'].score(test_d)
        
        defense_models = {'mae': self.mae, 'irm': self.irm}
        
        lambdas = [0.0, 10.0, 50.0, 100.0]
        results = [] 
        
        print(f"{'Lambda':<6} | {'Shift':<8} | {'WMAPE(%)':<8} | {'Ours(AUC)':<10} | {'PCA(AUC)':<10} | {'IF(AUC)':<10}")
        print("-" * 75)
        
        for lam in lambdas:
            # 1. Train Attack
            injector = BackTimeInjector()
            # Reduce epochs for speed in demo, but keep 20 for quality
            injector.train_attack(victim, atk_loader, epochs=20, defense_models=defense_models, lambda_defense=lam, 
                                  verbose=False, augmenter=self.augmenter, data_mean=self.data_mean, data_std=self.data_std) # Keep verbose=False
            
            # 2. Inject
            t_adv = injector.inject_attack(t_test, augmenter=self.augmenter, data_mean=self.data_mean, data_std=self.data_std)
            x_adv = t_adv.cpu().numpy()
            
            # 3. Victim Impact (RMSE/MAPE)
            with torch.no_grad():
                vm = torch.from_numpy(self.data_mean).float().to(self.device)
                vs = torch.from_numpy(self.data_std).float().to(self.device)
                
                pred_clean_norm = victim(t_test[:, :, 0, 0:1])
                pred_adv_norm = victim(t_adv[:, :, 0, 0:1])
                
                # Denorm Flow (Channel 0)
                # vm shape is (1,1,1,4) or similar. We need scalar for channel 0.
                mu_f = vm[..., 0].item() 
                std_f = vs[..., 0].item()
                
                pred_clean_raw = pred_clean_norm * std_f + mu_f
                pred_adv_raw = pred_adv_norm * std_f + mu_f
                
                metrics_clean = pred_clean_raw.cpu().numpy()
                metrics_adv = pred_adv_raw.cpu().numpy()
                
                rmse, wmape = self.compute_metrics(metrics_adv, metrics_clean)
                shift = np.mean(np.abs(pred_adv_norm.cpu().numpy() - pred_clean_norm.cpu().numpy()))

            # 4. AUC Evaluation
            scores_adv = self.detect_score(x_adv)
            y_true = np.concatenate([np.zeros(len(scores_clean)), np.ones(len(scores_adv))])
            
            try:
                auc_ours = roc_auc_score(y_true, np.concatenate([scores_clean, scores_adv]))
            except: auc_ours = 0.5
            
            # Baselines
            pca_adv = baselines['pca'].score(x_adv)
            try:
                auc_pca = roc_auc_score(y_true, np.concatenate([pca_clean, pca_adv]))
            except: auc_pca = 0.5
            
            if_adv = baselines['iforest'].score(x_adv)
            try:
                auc_if = roc_auc_score(y_true, np.concatenate([if_clean, if_adv]))
            except: auc_if = 0.5

            print(f"{lam:<6.1f} | {shift:<8.4f} | {wmape:<8.2f} | {auc_ours:<10.4f} | {auc_pca:<10.4f} | {auc_if:<10.4f}")
            results.append({'lambda': lam, 'wmape': wmape, 'auc_ours': auc_ours, 'auc_pca': auc_pca})

        print("\n[Final Verdict]")
        l_0 = results[0]
        l_10 = results[1]
        l_100 = results[-1]
        
        print(f"1. Impact Analysis:")
        print(f"   Unconstrained (L=0)   : WMAPE {l_0['wmape']:.2f}%")
        print(f"   Constrained   (L=100) : WMAPE {l_100['wmape']:.2f}%")
        
        print(f"2. Baseline Comparison (at Peak Impact L=10):")
        print(f"   Ours : {l_10['auc_ours']:.4f}")
        print(f"   PCA  : {l_10['auc_pca']:.4f}")
        
        if l_10['auc_ours'] >= l_10['auc_pca']:
            print("   -> Our defense Matched/Outperformed baselines.")
        else:
            print("   -> Baselines still competitive (Mean Fusion applied).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PEMS04", help="PEMS04, PEMS08, or PEMS03")
    args = parser.parse_args()
    
    pipeline = DefensePipeline(dataset=args.dataset)
    pipeline.run_eval()
