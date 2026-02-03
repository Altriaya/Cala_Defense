import torch
import torch.nn as nn
import torch.nn.functional as F

class InvariantGraphLearner(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_dim=64):
        """
        Args:
            num_nodes: Number of sensors/nodes.
            in_channels: Input dimension (Flow, Speed, Occ, Res).
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 1. Structure Learner (Node Embeddings -> Adjacency)
        # We learn two node embeddings E1, E2 to compute A = Softmax(E1 @ E2^T)
        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, hidden_dim).to(torch.float32))
        self.node_vec2 = nn.Parameter(torch.randn(num_nodes, hidden_dim).to(torch.float32))
        
        # 2. Graph Convolution (GCN/GAT) for Forecasting
        # Input: (B, T, N, C) -> We flatten T*C or use simpler mechanism
        # For simplicity: linear project input to hidden, then GCN, then pred.
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.gcn_weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, in_channels) # Predict next step or reconstruct
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_vec1)
        nn.init.xavier_uniform_(self.node_vec2)
        nn.init.xavier_uniform_(self.gcn_weight)

    def get_adjacency_matrix(self):
        # A = E1 @ E2^T
        # Saturation to make it sparse/sharper?
        adj = torch.mm(self.node_vec1, self.node_vec2.transpose(1, 0))
        adj = F.softmax(adj, dim=-1) # Row-normalized
        return adj

    def forward(self, x):
        """
        x: (B, T, N, C)
        We'll treat this as a simple graph regression:
        H = Linear(x)
        H' = A * H * W
        Out = Linear(H')
        
        For simplicity, let's process the LAST time step to predict next, 
        or process all steps independently (node-wise shared).
        Let's go with node-wise shared for all T.
        """
        B, T, N, C = x.shape
        adj = self.get_adjacency_matrix() # (N, N)
        
        # (B, T, N, C) -> (B*T, N, C)
        x_flat = x.view(-1, N, C)
        
        # 1. Projection
        h = self.input_proj(x_flat) # (Batch, N, Hidden)
        
        # 2. GCN
        # AXW
        # h: (Batch, N, Hid)
        # adj: (N, N)
        # AH = (Batch, N, N) x (Batch, N, Hid) -> (Batch, N, Hid) ?? No
        # (N, N) x (N, Batch*Hid) -> ...
        
        # Correct broadcast:
        # h = (Batch, N, Hid)
        # Support = h @ W -> (Batch, N, Hid)
        support = torch.matmul(h, self.gcn_weight)
        
        # Output = A @ Support
        # (N, N) @ (Batch, N, Hid) -> (Batch, N, Hid) ??
        # Use einsum for clarity: batch b, node i, node j, hidden f
        # adj: ij
        # support: bjf
        # out: bif
        output = torch.einsum('ij,bjf->bif', adj, support)
        
        output = F.relu(output)
        
        # 3. Final Pred
        pred = self.out_proj(output) # (Batch, N, C)
        
        pred = pred.view(B, T, N, C)
        return pred, adj

if __name__ == "__main__":
    model = InvariantGraphLearner(num_nodes=10, in_channels=4)
    x = torch.randn(2, 24, 10, 4)
    pred, adj = model(x)
    print(f"Pred shape: {pred.shape}")
    print(f"Adj shape: {adj.shape}")
