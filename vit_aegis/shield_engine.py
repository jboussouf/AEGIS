import torch
import torch.nn as nn

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
        return scores.view(original_shape[:-1]), projected

class LightweightAutoencoder(nn.Module):
    """
    Dimensionality reduction autoencoder for the AEGIS guardrail.
    Projects high-dimensional ViT embeddings to a lower-dimensional 
    manifold to isolate standard features from backdoor perturbations.
    """
    def __init__(self, input_dim=768, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class AEGIS_ViT_Shield:
    """
    AEGIS Core Logic for Vision Transformers.
    Monitors internal activations via forward hooks, projects them 
    onto the clean manifold, and computes geometric deviations.
    """
    def __init__(self, model, input_dim=768, latent_dim=64, device=None, method="pca"):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.method = method
        
        # Initialize lightweight mapping
        self.autoencoder = LightweightAutoencoder(input_dim, latent_dim).to(self.device)
        self.pca_shield = AEGIS_PCA_Shield(q=latent_dim, metric="cosine").to(self.device)
        self.centroid = None
        self.threshold = None
        
        # Internal state buffer to store hooked activations asynchronously
        self.activations = []
        self._register_hooks()

    def _get_ultimate_layer(self):
        # Robustly locate the transformer layers
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            return self.model.encoder.layer[-1]
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
            return self.model.encoder.layers[-1]
        if hasattr(self.model, 'blocks'):
            return self.model.blocks[-1]
        
        # Fallback to finding the largest ModuleList
        for module in self.model.modules():
            if isinstance(module, nn.ModuleList) and len(module) > 5:
                return module[-1]
        raise AttributeError("Could not locate the transformer layers for hooking.")

    def _register_hooks(self):
        """Registers a forward hook on the final transformer block."""
        def hook_fn(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Mean-pool across patch tokens (excluding the CLS token at index 0)
            pooled_activation = hidden_states[:, 1:, :].mean(dim=1)
            self.activations.append(pooled_activation)

        # Hook the ultimate encoder layer
        target_layer = self._get_ultimate_layer()
        target_layer.register_forward_hook(hook_fn)

    def fit_baseline(self, clean_activations):
        """
        Fits the clean baseline centroid on a trusted calibration set.
        Establishes the upper-bound threshold for normal variance.
        """
        if self.method == "pca":
            self.pca_shield.fit(clean_activations)
            with torch.no_grad():
                distances, _ = self.pca_shield.score(clean_activations)
            self.threshold = torch.quantile(distances, 0.95)
        else:
            self.autoencoder.eval() 
            with torch.no_grad():
                latents, _ = self.autoencoder(clean_activations)
                
                # Compute the geometric centroid of clean latents
                self.centroid = latents.mean(dim=0, keepdim=True)
                
                # GPU-native cosine distances to establish empirical threshold (95th percentile)
                distances = 1 - torch.nn.functional.cosine_similarity(latents, self.centroid, dim=-1)
                self.threshold = torch.quantile(distances, 0.95)

    def analyze_pass(self):
        """
        Analyzes the hooked activations from the most recent forward pass.
        Guarantees NO .cpu() calls to eliminate host-device synchronization latency.
        """
        if not self.activations:
            raise ValueError("No activations found. Execute a model forward pass first.")
            
        act = self.activations.pop(0) # FIFO pop
        
        with torch.no_grad():
            if self.method == "pca":
                distances, latents = self.pca_shield.score(act)
            else:
                latents, _ = self.autoencoder(act)
                # 1 - Cosine Similarity yields Cosine Distance
                distances = 1 - torch.nn.functional.cosine_similarity(latents, self.centroid, dim=-1)
            
        is_blocked = distances > self.threshold
        return distances, is_blocked, latents
