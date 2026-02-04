import torch
import numpy as np
import sys
import os

# Set up paths to import from BackTime-main
curr_dir = os.path.dirname(os.path.abspath(__file__))
# attack_dir = os.path.join(curr_dir, 'attack', 'BackTime-main')
# We need to point to where 'trigger.py' is. 
# d:\DLProjectHome\Cala_Defense\attack\BackTime-main
repo_dir = os.path.join(curr_dir, 'attack', 'BackTime-main')
sys.path.append(repo_dir)

try:
    from trigger import TgrGCN
except ImportError as e:
    print(f"[AttackAdapter] Error importing BackTime modules: {e}")
    # Fallback or exit? For now, assume it works if path is right.

class MockConfig:
    def __init__(self, seq_len=24, hidden_dim=64, nodes=10):
        # We split seq_len (24) into History (12) + Trigger (12)
        self.bef_tgr_len = 12 
        self.hidden_dim = hidden_dim
        self.trigger_len = 12 
        self.epsilon = 5.0 # Increased from 0.5 to 5.0 to allow strong attacks
        self.batch_size = 1

class BackTimeInjector:
    def __init__(self, num_nodes=10, seq_len=24):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nodes = num_nodes
        self.seq_len = seq_len
        self.config = MockConfig(seq_len=seq_len, nodes=num_nodes)
        
        # We need "sim_feats" for TgrGCN (node features). 
        # Usually these are FFT features of the time series.
        # For mock, we can just use random embeddings.
        # Shape: (num_nodes, feature_dim)
        sim_feats = np.random.randn(num_nodes, 16)
        
        # "atk_vars": which nodes to attack. Let's attack Node 0.
        self.atk_vars = torch.tensor([0], device=self.device).long()
        
        # Initialize Generator
        self.generator = TgrGCN(self.config, sim_feats, self.atk_vars, device=str(self.device).split(':')[0])
        self.generator.to(self.device)
        
        # Optimizer for Generator
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.05)

    def enforce_physics(self, batch_norm, augmenter, mean, std):
        """
        Re-calculates the Physics Residual channel (Channel 3) based on modified Flow/Speed/Occ.
        Differentiable.
        """
        # batch_norm: (..., 4)
        # 1. Denormalize First 3 Channels (Flow, Occ, Speed)
        raw_core = batch_norm[..., :3] * std[..., :3] + mean[..., :3]
        
        # 2. Augment (Calculate Residual)
        aug_raw = augmenter.augment(raw_core)
        
        # 3. Normalize
        norm_new = (aug_raw - mean) / std
        return norm_new

    def train_attack(self, victim_model, data_loader, epochs=100, defense_models=None, lambda_defense=0.0, 
                     verbose=True, augmenter=None, data_mean=None, data_std=None):
        """
        Optimize the generator to fool the victim model AND evade defense.
        """
        mode_str = "Adaptive" if defense_models else "Standard"
        if verbose:
            print(f"[Attack] Training ({mode_str} L={lambda_defense}) Trigger Generator for {epochs} epochs...")
        self.generator.train()
        victim_model.train()
        
        if defense_models:
            if 'mae' in defense_models: defense_models['mae'].eval()
            if 'irm' in defense_models: defense_models['irm'].eval()

        target_val = 5.0 
        target_tensor = torch.full((1, 12), target_val).to(self.device)
        
        # Convert stats to tensor if needed
        if data_mean is not None:
             t_mean = torch.from_numpy(data_mean).float().to(self.device)
             t_std = torch.from_numpy(data_std).float().to(self.device)
        
        for epoch in range(epochs):
            total_loss = 0
            total_atk_loss = 0
            total_def_loss = 0
            
            for batch_idx, batch in enumerate(data_loader):
                # batch: (B, T, N, C) - Normalized
                
                # 1. Prepare Attack Input (Node 0)
                context = batch[:, :12, 0, 0] 
                
                # 2. Generate Trigger
                trigger_seq, perturb = self.generator(context)
                trigger_vals = trigger_seq.view(batch.shape[0], 12)
                
                # 3. Inject Trigger 
                attacked_batch = batch.clone()
                attacked_batch[:, 12:, 0, 0] = trigger_vals
                
                # --- PHYSICS CONSISTENCY ---
                if augmenter and data_mean is not None:
                    attacked_batch = self.enforce_physics(attacked_batch, augmenter, t_mean, t_std)
                # ---------------------------
                
                # 4. Victim Forward
                attacked_input_flow = attacked_batch[:, :, 0, 0]
                victim_in = attacked_input_flow.unsqueeze(-1)
                pred = victim_model(victim_in)
                
                # 5. Losses
                loss_attack = torch.mean((pred - target_tensor) ** 2)
                loss_norm = torch.mean(perturb ** 2)
                
                loss = loss_attack + 0.05 * loss_norm
                
                # 6. Adaptive Defense Loss
                if defense_models and lambda_defense > 0:
                    loss_evade = 0
                    if 'mae' in defense_models:
                        mae = defense_models['mae']
                        B, T, N, C = attacked_batch.shape
                        flat_input = attacked_batch.transpose(1, 2).reshape(B*N, T, C)
                        mae_pred, mask, _ = mae(flat_input)
                        loss_mae = torch.mean((mae_pred - flat_input)**2)
                        loss_evade += loss_mae
                        
                    if 'irm' in defense_models:
                        irm = defense_models['irm']
                        irm_pred, _ = irm(attacked_batch)
                        loss_irm = torch.mean((irm_pred - attacked_batch)**2)
                        loss_evade += loss_irm
                        
                    loss += lambda_defense * loss_evade
                    total_def_loss += loss_evade.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_atk_loss += loss_attack.item()
            
            if verbose and epoch % 10 == 0:
                def_msg = f" | DefLoss: {total_def_loss:.2f}" if lambda_defense > 0 else ""
                print(f"  [Attack Epoch {epoch}] Total: {total_loss:.2f} | AtkLoss: {total_atk_loss:.2f}{def_msg}")
                
        self.generator.eval()

    def inject_attack(self, data_norm, augmenter=None, data_mean=None, data_std=None):
        """
        Injects BackTime trigger into the data.
        Args:
            data_norm: (1, T, N, C) Normalized data.
        Returns:
            attacked_data: (1, T, N, C)
        """
        # B=1 or N
        context = data_norm[:, :12, 0, 0] 
        with torch.no_grad():
            trigger_seq, _ = self.generator(context) 
            
        trigger_vals = trigger_seq.view(data_norm.shape[0], 12)
        
        # Inject
        attacked_data = data_norm.clone()
        attacked_data[:, 12:, 0, 0] = trigger_vals
        
        # Update Physics
        if augmenter and data_mean is not None:
             if isinstance(data_mean, np.ndarray):
                 t_mean = torch.from_numpy(data_mean).float().to(self.device)
                 t_std = torch.from_numpy(data_std).float().to(self.device)
             else:
                 t_mean = data_mean
                 t_std = data_std
                 
             attacked_data = self.enforce_physics(attacked_data, augmenter, t_mean, t_std)
        
        return attacked_data

if __name__ == "__main__":
    injector = BackTimeInjector()
    dummy = torch.randn(1, 24, 10, 4).cuda()
    out = injector.inject_attack(dummy)
    print(f"Injected shape: {out.shape}")
    print("Diff in Flow:", (out - dummy)[0, 12:, 0, 0])
