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

    def train_attack(self, victim_model, data_loader, epochs=100, defense_models=None, lambda_defense=0.0):
        """
        Optimize the generator to fool the victim model AND evade defense.
        
        Args:
            defense_models: dict {'mae': model, 'irm': model} (Optional)
            lambda_defense: weight for defense evasion loss
        """
        mode_str = "Adaptive" if defense_models else "Standard"
        print(f"[Attack] Training ({mode_str}) Trigger Generator for {epochs} epochs...")
        self.generator.train()
        victim_model.train()
        
        # If adaptive, set defense models to eval
        if defense_models:
            if 'mae' in defense_models: defense_models['mae'].eval()
            if 'irm' in defense_models: defense_models['irm'].eval()

        target_val = 5.0 
        target_tensor = torch.full((1, 12), target_val).to(self.device)
        
        for epoch in range(epochs):
            total_loss = 0
            total_atk_loss = 0
            total_def_loss = 0
            
            for batch in data_loader:
                # batch: (B, T, N, C) - Normalized
                
                # 1. Prepare Attack Input (Node 0)
                context = batch[:, :12, 0, 0] 
                
                # 2. Generate Trigger
                trigger_seq, perturb = self.generator(context)
                trigger_vals = trigger_seq.view(-1, 12)
                
                # 3. Inject Trigger 
                # We need full batch for Defense models
                attacked_batch = batch.clone()
                # Inject into Node 0, Channel 0, steps 12:24
                attacked_batch[:, 12:, 0, 0] = trigger_vals
                
                # 4. Victim Forward (uses Node 0 Flow)
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
                    
                    # MAE Loss
                    if 'mae' in defense_models:
                        mae = defense_models['mae']
                        # MAE expects (N*B, T, C). Flatten nodes.
                        B, T, N, C = attacked_batch.shape
                        flat_input = attacked_batch.transpose(1, 2).reshape(B*N, T, C)
                        # We only care about Node 0, but MAE processes all. 
                        # Optimization is efficiently propagated through Node 0 part.
                        
                        # MAE Forward
                        # We need 'pred' from MAE.
                        mae_pred, mask, _ = mae(flat_input)
                        # Loss is MSE(pred, input)
                        # We want to Minimize this reconstruction error (to look "Normal")
                        loss_mae = torch.mean((mae_pred - flat_input)**2)
                        loss_evade += loss_mae
                        
                    # IRM Loss
                    if 'irm' in defense_models:
                        irm = defense_models['irm']
                        # IRM expects (B, T, N, C) -> (pred, adj)
                        irm_pred, _ = irm(attacked_batch)
                        loss_irm = torch.mean((irm_pred - attacked_batch)**2)
                        loss_evade += loss_irm
                        
                    # Add to total loss
                    loss += lambda_defense * loss_evade
                    total_def_loss += loss_evade.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_atk_loss += loss_attack.item()
            
            if epoch % 20 == 0:
                def_msg = f" | DefLoss: {total_def_loss:.2f}" if lambda_defense > 0 else ""
                print(f"  [Attack Epoch {epoch}] Total: {total_loss:.2f} | AtkLoss: {total_atk_loss:.2f}{def_msg}")
                
        self.generator.eval()

    def inject_attack(self, data_norm):
        """
        Injects BackTime trigger into the data.
        Args:
            data_norm: (1, T, N, C) Normalized data.
        Returns:
            attacked_data: (1, T, N, C)
        """
        # Data is (B, T, N, C).
        # TgrGCN expects input (B, input_dim). 
        # input_dim = bef_tgr_len.
        
        # Let's say we attack the last 12 steps (trigger_len).
        # We use the first 12 steps as context (bef_tgr_len).
        
        # B=1
        context = data_norm[:, :12, 0, 0] # Use Node 0, Channel 0 (Flow), first 12 steps
        # Shape (1, 12).
        
        # Generator forward
        # x: (batch, input_dim) -> (batch, trigger_len)
        with torch.no_grad():
            # TgrGCN output_dim is trigger_len.
            trigger_seq, _ = self.generator(context) 
            
        trigger_vals = trigger_seq.view(1, 12)
        
        # Scale no longer needed if we trained it in the normalized space!
        # It should have learned the scale required to fool the victim.
        
        # Inject
        attacked_data = data_norm.clone()
        # Inject into Node 0, Channel 0, steps 12:24
        attacked_data[:, 12:, 0, 0] = trigger_vals
        
        return attacked_data

if __name__ == "__main__":
    injector = BackTimeInjector()
    dummy = torch.randn(1, 24, 10, 4).cuda()
    out = injector.inject_attack(dummy)
    print(f"Injected shape: {out.shape}")
    print("Diff in Flow:", (out - dummy)[0, 12:, 0, 0])
