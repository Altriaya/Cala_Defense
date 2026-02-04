import torch
import numpy as np
import sys
import os

# Path setup
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)

# Utils
from utils.greenshields import GreenshieldsProfiler
from utils.physical_augmenter import PhysicalAugmenter
from utils.environment_splitter import TimeBasedSplitter

# Models & Trainers
from consistency.mae_model import PhysicsMAE
from consistency.train_mae import MAETrainer
from invariance.graph_learner import InvariantGraphLearner
from invariance.train_irm import IRMTrainer
from models import TimesNetForecaster
from attack_adapter import BackTimeInjector
from detection.scorer import AnomalyScorer

from sklearn.metrics import roc_auc_score, roc_curve

class DefensePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Initializing on {self.device}...")
        self.data_path = 'd:/DLProjectHome/Cala_Defense/data/PEMS04/PEMS04.npz'
        
        # State
        self.mae = None
        self.irm = None
        self.scorer = AnomalyScorer()
        self.data_mean = None
        self.data_std = None
        self.augmenter = None
        
    def prepare_data(self):
        print("[Phase 0] Preparing Real PEMS04 Data...")
        try:
            # 1. Load PEMS04
            data_raw = np.load(self.data_path)['data'] # (16992, 307, 3)
            # PEMS04 (T, N, C). Channels: Flow, Occ, Speed?
            # Standard PEMS sequence usually: flow, occ, speed.
            # Profiler will confirm constraints.
            
            # Slice for demo speed (use first 2000 steps, ~1 week)
            # Full dataset is too large for quick demo training
            demo_steps = 4000 
            data_slice = data_raw[:demo_steps, :10, :] # Use first 10 nodes for speed
            
            print(f"  Data loaded: {data_slice.shape}")

            # 2. Physics Profiling (Greenshields)
            print("  [Physics] Fitting Greenshields Model...")
            # We need T*N samples.
            profiler = GreenshieldsProfiler(data_array=data_slice)
            v_f, k_j = profiler.fit_physics()
            
            # 3. Augment
            self.augmenter = PhysicalAugmenter(v_f=v_f, k_j=k_j)
            aug_data = self.augmenter.augment(data_slice) # (T, N, 4)
            
            # 4. Normalize
            self.data_mean = np.mean(aug_data, axis=(0, 1), keepdims=True)
            self.data_std = np.std(aug_data, axis=(0, 1), keepdims=True)
            self.data_std[self.data_std < 1e-6] = 1.0
            
            norm_data = (aug_data - self.data_mean) / self.data_std
            
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
            sys.exit(1)

    def train_defense(self, train_data, raw_sequence_slice):
        print("\n[Phase 1] Training Defense Models (Scientific)...")
        
        # 1. Train MAE
        print("  [1.1] Training Physics-MAE...")
        self.mae = PhysicsMAE(seq_len=24, in_channels=4, patch_size=1, mask_ratio=0.4).to(self.device)
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
        print("  [1.2] Training Invariant Graph Learner (IRM)...")
        
        splitter = TimeBasedSplitter(steps_per_day=288)
        
        envs = [[], [], []] # AM, PM, Off
        for i in range(len(train_data)):
            t_start = i * 1 # Stride 1
            env_id = splitter.get_env_id(t_start)
            envs[env_id].append(train_data[i])
            
        env_tensors = [torch.from_numpy(np.array(e)).float().to(self.device) for e in envs if len(e) > 0]
        print(f"    IRM Environments: {[e.shape[0] for e in env_tensors]}")
        
        self.irm = InvariantGraphLearner(num_nodes=N, in_channels=4).to(self.device)
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

    def calibrate(self, val_data):
        print("\n[Phase 2] Calibrating Scorer (Val Data)...")
        self.mae.eval()
        self.irm.eval()
        losses = {'mae': [], 'irm': []}
        
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
            
            stats = self.scorer.stats
            z_mae = (loss_mae - stats['mae']['mu']) / stats['mae']['sigma']
            z_irm = (loss_irm - stats['irm']['mu']) / stats['irm']['sigma']
            
            scores = (z_mae + z_irm) / 2.0
            return scores

    def run_eval(self):
        # 1. Prep (Sliding Window)
        data, raw_slice = self.prepare_data()
        
        n = len(data)
        train_d = data[:int(0.5*n)]
        val_d = data[int(0.5*n):int(0.75*n)]
        test_d = data[int(0.75*n):]
        
        # 2. Train Defense
        self.train_defense(train_d, raw_slice)
        
        # 3. Calibrate
        self.calibrate(val_d)
        
        # 4. Train Victim (Mock for speed)
        print("\n[Phase 3] Training Victim Model (TimesNet)...")
        # Define Simple Victim locally if not imported
        victim = TimesNetForecaster(in_dim=1, out_dim=12).to(self.device)
        optimizer = torch.optim.Adam(victim.parameters(), lr=0.01)
        # Train on train_d (Flow channel)
        v_train_in = torch.from_numpy(train_d[:500, :24, 0, 0:1]).float().to(self.device) # Subset for speed
        victim.train()
        for e in range(30):
             p = victim(v_train_in[:,:12])
             l = torch.nn.functional.mse_loss(p, v_train_in[:,12:,0])
             optimizer.zero_grad(); l.backward(); optimizer.step()
        print("  Victim Trained.")
        
        # 5. Attack & Lambda Sweep
        print("\n[Phase 4] Verifying Pareto Frontier (Lambda Sweep)...")
        
        atk_subset_indices = np.random.choice(len(train_d), size=min(len(train_d), 500), replace=False)
        atk_subset = train_d[atk_subset_indices]
        atk_loader = torch.utils.data.DataLoader(torch.from_numpy(atk_subset).float().to(self.device), batch_size=32, shuffle=True)
        
        t_test = torch.from_numpy(test_d).float().to(self.device)
        scores_clean = self.detect_score(test_d)
        
        defense_models = {'mae': self.mae, 'irm': self.irm}
        
        lambdas = [0.0, 10.0, 50.0, 100.0]
        results = [] 
        
        print(f"{'Lambda':<8} | {'AUC':<10} | {'Shift':<15} | {'Status'}")
        print("-" * 55)
        
        for lam in lambdas:
            injector = BackTimeInjector()
            injector.train_attack(victim, atk_loader, epochs=20, defense_models=defense_models, lambda_defense=lam, 
                                  verbose=False, augmenter=self.augmenter, data_mean=self.data_mean, data_std=self.data_std)
            
            t_adv = injector.inject_attack(t_test, augmenter=self.augmenter, data_mean=self.data_mean, data_std=self.data_std)
            
            with torch.no_grad():
                pred_clean = victim(t_test[:, :, 0, 0:1]).mean().item()
                pred_adv = victim(t_adv[:, :, 0, 0:1]).mean().item()
                shift = abs(pred_adv - pred_clean)
            
            x_adv = t_adv.cpu().numpy()
            scores_adv = self.detect_score(x_adv)
            
            y_true = np.concatenate([np.zeros_like(scores_clean), np.ones_like(scores_adv)])
            y_score = np.concatenate([scores_clean, scores_adv])
            try:
                auc = roc_auc_score(y_true, y_score)
            except:
                auc = 0.5
            
            status = "Stealthy" if auc < 0.65 else "Detected"
            if shift < 1.0: status += " & Weak"
            
            print(f"{lam:<8.1f} | {auc:<10.4f} | {shift:<15.4f} | {status}")
            results.append({'lambda': lam, 'auc': auc, 'shift': shift})

        print("\n[Final Verdict]")
        l_0 = results[0]
        l_100 = results[-1]
        
        print(f"Standard Attack (L=0)   -> AUC: {l_0['auc']:.4f}")
        print(f"Extreme Constraint (L=100) -> AUC: {l_100['auc']:.4f}, Shift: {l_100['shift']:.4f}")
        
        if l_0['auc'] > 0.70:
            print("1. Defense is EFFECTIVE against naive attacks.")
            if l_100['auc'] < 0.65 and l_100['shift'] < l_0['shift'] * 0.6:
                print("2. Pareto Trade-off CONFIRMED: High stealth forces significantly reduced Impact.")
                print("   The defense successfully forces the attacker into a dilemma.")
            elif l_100['auc'] > 0.70:
                print("2. Defense is ROBUST: Even L=100 cannot evade detection.")
            else:
                 print("2. Trade-off Unclear. Check penalties.")
        else:
            print("1. Defense Failed Base Case.")

if __name__ == "__main__":
    pipeline = DefensePipeline()
    pipeline.run_eval()
