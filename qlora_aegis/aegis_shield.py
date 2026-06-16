import torch
import torch.nn as nn
import logging

logger = logging.getLogger("AEGIS_QLoRA_PoC.Shield")

class AEGIS_PCA_Shield(nn.Module):
    def __init__(self, q=64, metric="cosine"):
        super().__init__()
        self.q = q
        self.metric = metric
        self.register_buffer('clean_mean', None)
        self.register_buffer('V', None)

    def fit(self, clean_activations):
        if clean_activations.dim() > 2:
            clean_activations = clean_activations.view(-1, clean_activations.size(-1))
        mean = clean_activations.mean(dim=0)
        self.register_buffer('clean_mean', mean)
        centered_acts = clean_activations - self.clean_mean
        q = min(self.q, clean_activations.size(0))
        U, S, V = torch.pca_lowrank(centered_acts, q=q)
        self.register_buffer('V', V)

    def score(self, new_activations):
        original_shape = new_activations.shape
        if new_activations.dim() > 2:
            new_activations = new_activations.view(-1, new_activations.size(-1))
        centered = new_activations - self.clean_mean
        projected = torch.matmul(centered, self.V)
        reconstructed = torch.matmul(projected, self.V.t()) + self.clean_mean
        if self.metric == "cosine":
            scores = 1.0 - torch.nn.functional.cosine_similarity(new_activations, reconstructed, dim=-1)
        else:
            scores = torch.norm(new_activations - reconstructed, dim=-1)
        return scores.view(original_shape[:-1])


class AEGIS_Shield(nn.Module):
    """
    AEGIS (Activation Evaluation & Guardrail for Inference Security) Shield.
    A lightweight, GPU-native autoencoder to monitor internal activation manifolds
    and detect latent-space shifts induced by adversarial QLoRA adapters.
    """
    def __init__(self, input_dim: int = 4096, latent_dim: int = 64, method: str = "pca"):
        super().__init__()
        self.method = method
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, input_dim)
        )
        
        self.pca_shield = AEGIS_PCA_Shield(q=latent_dim, metric="cosine")
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        self.threshold = 0.0

    def fit_pca(self, clean_activations: torch.Tensor):
        self.pca_shield.fit(clean_activations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.method == "pca":
                return self.pca_shield.score(x)
            else:
                reconstructed = self(x)
                sim = self.cosine_sim(x, reconstructed)
                return 1.0 - sim

    def calibrate(self, clean_activations: torch.Tensor, percentile: float = 0.95):
        logger.info("Calibrating AEGIS Shield on clean activation manifold...")
        scores = self.compute_anomaly_score(clean_activations)
        self.threshold = torch.quantile(scores, percentile).item()
        logger.info(f"AEGIS Threshold calibrated at {percentile*100}th percentile: {self.threshold:.4f}")
