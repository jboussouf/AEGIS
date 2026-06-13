import numpy as np
from sklearn.ensemble import IsolationForest
import torch

class ActivationShield:
    def __init__(self, model, target_layers=[2, 12, 24]): # Example for 32-layer model
        self.model = model
        self.target_layers = target_layers
        self.hooks = []
        self.activation_buffer = {}
        # contamination=0.1 means we expect ~10% of baseline samples could be
        # borderline; more robust than 0.01 on small fitting corpora.
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self._register_hooks()

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            # output is typically a tuple, we want the hidden states tensor
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Pool sequence dimension (e.g., mean pooling over sequence length)
            pooled_activation = hidden_states.mean(dim=1).detach().cpu().numpy()
            self.activation_buffer[layer_name] = pooled_activation
        return hook

    def _register_hooks(self):
        # Note: Accessing model.model.layers is specific to certain architectures like Llama
        # Adapt this based on the specific model architecture being tested
        layers = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h # GPT-2
            
        if layers is None:
            print("Warning: Could not automatically find layers. Hooks not registered.")
            return

        for i, layer in enumerate(layers):
            if i in self.target_layers:
                hook = layer.register_forward_hook(self._hook_fn(f"layer_{i}"))
                self.hooks.append(hook)

    def fit_baseline(self, dataloader):
        """Fit the Isolation Forest on known clean activations."""
        print("Fitting baseline Isolation Forest...")
        baseline_activations = []
        
        for batch in dataloader:
            self.activation_buffer.clear()
            # Assuming batch is a dict of tensors
            with torch.no_grad():
                _ = self.model(**batch)
            
            # Concatenate features from target layers
            try:
                current_features = np.concatenate([self.activation_buffer[f"layer_{i}"] for i in self.target_layers], axis=1)
                baseline_activations.append(current_features)
            except KeyError:
                print("Error: Missing activations. Check if hooks are firing.")
                
        if baseline_activations:
            all_features = np.vstack(baseline_activations)
            self.anomaly_detector.fit(all_features)
            print("Baseline fitted successfully.")

    def monitor_stream(self, input_batch):
        """Real-time inference monitoring."""
        self.activation_buffer.clear()
        
        # Forward pass populates the buffer via hooks.
        # Use duck-typing (hasattr 'items') instead of isinstance(dict) so that
        # HuggingFace BatchEncoding objects—which are NOT plain dicts—are also
        # correctly unpacked with **kwargs.
        if hasattr(input_batch, 'items'):
            _ = self.model(**input_batch)
        else:
            _ = self.model(input_batch)
        
        # Aggregate activations across hooked layers
        try:
            current_features = np.concatenate(
                [self.activation_buffer[f"layer_{i}"] for i in self.target_layers],
                axis=1
            )
        except KeyError:
            print("Error: Missing activations. Hooks did not capture target layers.")
            return False
        
        # IsolationForest must be fitted on baseline data before predict.
        # If not yet fitted, flag as unknown (False = not detected).
        try:
            anomaly_score = self.anomaly_detector.predict(current_features)
            is_attack = anomaly_score[0] == -1
        except Exception:
            print("Warning: Anomaly detector not yet fitted. Call fit_baseline() first. Defaulting to no detection.")
            is_attack = False
        
        return is_attack

    def score_stream(self, input_batch):
        """
        Return a continuous anomaly score for the input batch.

        IsolationForest.decision_function() returns scores where:
          - More negative  → more anomalous (likely attack)
          - Around 0 / positive → normal

        We negate the score so that HIGHER values signal greater suspicion,
        which is the convention expected by roc_auc_score.
        """
        self.activation_buffer.clear()

        if hasattr(input_batch, 'items'):
            _ = self.model(**input_batch)
        else:
            _ = self.model(input_batch)

        try:
            current_features = np.concatenate(
                [self.activation_buffer[f"layer_{i}"] for i in self.target_layers],
                axis=1
            )
        except KeyError:
            return 0.0  # neutral score when hooks miss

        try:
            # decision_function returns shape (n_samples,); negate so higher = worse
            raw_score = self.anomaly_detector.decision_function(current_features)
            return float(-raw_score[0])
        except Exception:
            return 0.0
