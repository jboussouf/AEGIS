import torch
import torch.nn as nn
import torch.nn.functional as F

class AEGIS_PCA_Shield(nn.Module):
    """
    Principal Component Analysis (PCA) Shield for Project AEGIS.
    A training-free, deterministic guardrail that leverages low-rank
    approximation to model the clean activation manifold.
    """
    def __init__(self, q=64, metric="cosine"):
        super().__init__()
        self.q = q
        self.metric = metric
        
        # Registering buffers ensures they automatically map to the GPU 
        # when the shield module is moved via .to(device)
        self.register_buffer('clean_mean', None)
        self.register_buffer('V', None)

    def fit(self, clean_activations):
        """
        Fits the PCA shield to a hold-out set of clean activations.
        This calculates the centroid (mean) and the principal components natively on the GPU.
        
        Args:
            clean_activations (torch.Tensor): Shape (N, D)
        """
        if clean_activations.dim() > 2:
            clean_activations = clean_activations.view(-1, clean_activations.size(-1))
            
        # Compute and store the clean centroid
        mean = clean_activations.mean(dim=0)
        self.register_buffer('clean_mean', mean)
        
        # Center the data around the clean manifold
        centered_acts = clean_activations - self.clean_mean
        
        # Use PyTorch's native GPU-accelerated low-rank PCA
        # U: (N, q), S: (q,), V: (D, q)
        U, S, V = torch.pca_lowrank(centered_acts, q=self.q)
        
        # Store the principal components
        self.register_buffer('V', V)

    def score(self, new_activations):
        """
        Scoring phase for prefill-only anomaly detection.
        Executes without ANY CPU-GPU synchronization (.item(), .cpu() are strictly omitted).
        
        Args:
            new_activations (torch.Tensor): Shape (..., D)
            
        Returns:
            scores (torch.Tensor): The anomaly scores.
        """
        original_shape = new_activations.shape
        if new_activations.dim() > 2:
            new_activations = new_activations.view(-1, new_activations.size(-1))
            
        # 1. Center the new activations against the clean centroid
        centered = new_activations - self.clean_mean
        
        # 2. Project onto the top q principal components
        # (N, D) x (D, q) -> (N, q)
        projected = torch.matmul(centered, self.V)
        
        # 3. Reconstruct them back to the original space
        # (N, q) x (q, D) -> (N, D)
        reconstructed = torch.matmul(projected, self.V.t()) + self.clean_mean
        
        # 4. Calculate GPU-native Distance / Reconstruction Error
        if self.metric == "cosine":
            # Cosine Distance (1 - Cosine Similarity) between the input and its PCA reconstruction
            scores = 1.0 - F.cosine_similarity(new_activations, reconstructed, dim=-1)
        else:
            # L2 Reconstruction Error
            scores = torch.norm(new_activations - reconstructed, dim=-1)
            
        return scores.view(original_shape[:-1])
