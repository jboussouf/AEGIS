import torch
import torch.nn as nn
import logging

logger = logging.getLogger("AEGIS_QLoRA_PoC.Shield")

class AEGIS_Shield(nn.Module):
    """
    AEGIS (Activation Evaluation & Guardrail for Inference Security) Shield.
    A lightweight, GPU-native autoencoder to monitor internal activation manifolds
    and detect latent-space shifts induced by adversarial QLoRA adapters.
    """
    def __init__(self, input_dim: int = 4096, latent_dim: int = 64):
        super().__init__()
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
        
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        self.threshold = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reconstructed = self(x)
            sim = self.cosine_sim(x, reconstructed)
            return 1.0 - sim

    def calibrate(self, clean_activations: torch.Tensor, percentile: float = 0.95):
        logger.info("Calibrating AEGIS Shield on clean activation manifold...")
        scores = self.compute_anomaly_score(clean_activations)
        self.threshold = torch.quantile(scores, percentile).item()
        logger.info(f"AEGIS Threshold calibrated at {percentile*100}th percentile: {self.threshold:.4f}")
