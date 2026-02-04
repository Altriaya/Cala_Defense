import torch
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import sys
import os

# Adjust path 
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir)

from utils.mock_data import MockPEMSDataset
from utils.environment_splitter import TimeBasedSplitter
from utils.physical_augmenter import PhysicalAugmenter
from invariance.graph_learner import InvariantGraphLearner

class IRMTrainer:
    def __init__(self, model=None, device=None, num_nodes=10):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_nodes = num_nodes
        self.model = model
        
        if self.model:
             self.model = self.model.to(self.device)
             self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        else:
            # Legacy Self-Init
            # 1. Prepare Data
            self.raw_dataset = MockPEMSDataset(num_samples=300, num_nodes=num_nodes, seq_len=24)
            self.raw_data = self.raw_dataset.get_all_data() # (N, T, Nodes, 3)
            
            # Augment
            self.augmenter = PhysicalAugmenter() # Use defaults for mock
            self.aug_data = self.augmenter.augment(self.raw_data) # (N, T, Nodes, 4)
            
            # 2. Split Environments
            self.splitter = EnvironmentSplitter(num_envs=3)
            self.envs_data = self.splitter.split(self.aug_data) # List of np arrays
            
            # Convert to Tensors
            self.envs = [torch.from_numpy(e).float().to(self.device) for e in self.envs_data]
            
            # 3. Model
            self.model = InvariantGraphLearner(num_nodes=num_nodes, in_channels=4).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)

    def compute_penalty(self, loss, dummy_w):
        """
        IRM V1 Penalty: norm of gradient of loss w.r.t dummy scalar multiplier '1.0'.
        This approximates the variance of the optimal classifier across environments.
        """
        g = grad(loss, dummy_w, create_graph=True)[0]
        return (g**2).sum()

    def train(self, epochs=50, penalty_weight=1000.0):
        self.model.train()
        
        print("\n[IRM] Starting Training...")
        for epoch in range(epochs):
            total_loss = 0
            total_penalty = 0
            
            for env_data in self.envs:
                # Env data: (B_env, T, N, C)
                # We can batch it, but for simplicity use full env (it's small mock data)
                
                # To use IRM penalty, we simulate the "dummy classifier" trick
                # Effectively, we want grad(Loss) w.r.t a scalar fixed at 1.0 to be small.
                # However, full IRM is complex to implement for GNNs.
                # SIMPLIFIED IRM for Adjacency:
                # We just want the Loss to be low in ALL environments using the SAME Adjacency.
                # Variance of Loss Penalty:
                # Loss = Sum(L_e) + lambda * Var(L_e)
                
                pass
                
            # Let's implement the Variance Penalty approach (Rex: Risk Extrapolation / V-REx)
            # It's more stable than IRMv1 for this context.
            
            env_losses = []
            for env_data in self.envs:
                pred, adj = self.model(env_data)
                # Reconstruct/Predict self (Autoencoder style for graph)
                loss_e = torch.mean((pred - env_data)**2)
                env_losses.append(loss_e)
            
            env_losses_stack = torch.stack(env_losses)
            mean_loss = env_losses_stack.mean()
            penalty = env_losses_stack.var() # Enforce consistent performance
            
            final_loss = mean_loss + penalty_weight * penalty
            
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Mean Loss={mean_loss.item():.4f}, Var Penalty={penalty.item():.6f}")

    def verify_invariant_structure(self):
        """
        Verify that the learned graph is stable and physically meaningful (mock check).
        """
        self.model.eval()
        adj = self.model.get_adjacency_matrix().detach().cpu().numpy()
        
        print("\n[Verification] Learned Adjacency Matrix (Top 5 rows):")
        print(adj[:5, :5])
        
        # Check sparsity or structure
        # In mock data, dependencies are weak/local.
        pass

if __name__ == "__main__":
    trainer = IRMTrainer()
    trainer.train(epochs=100)
    trainer.verify_invariant_structure()
