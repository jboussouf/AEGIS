import torch
import logging
from typing import List

logger = logging.getLogger("AEGIS_Real_QLoRA.HookManager")

class HookManager:
    def __init__(self, target_layers: List[int]):
        self.target_layers = target_layers
        self.activations = {}
        self.hooks = []

    def get_hook(self, layer_idx: int):
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Only capture activations during the prefill phase (seq_len > 1) to save latency
            if hidden_states.shape[1] > 1:
                pooled = hidden_states.mean(dim=1).detach()
                if layer_idx not in self.activations:
                    self.activations[layer_idx] = []
                self.activations[layer_idx].append(pooled)
        return hook

    def register_hooks(self, model):
        for layer_idx in self.target_layers:
            try:
                # Handle PEFT wrapped model specifically for Qwen architecture
                layer = model.base_model.model.model.layers[layer_idx]
                self.hooks.append(layer.register_forward_hook(self.get_hook(layer_idx)))
            except AttributeError as e:
                logger.error(f"Failed to hook layer {layer_idx}. Verify model architecture.")
        logger.info(f"Registered AEGIS hooks on layers: {self.target_layers}")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def clear_activations(self):
        self.activations.clear()

    def get_concatenated_activations(self) -> torch.Tensor:
        tensors = []
        for idx in self.target_layers:
            if idx in self.activations and len(self.activations[idx]) > 0:
                tensors.append(self.activations[idx][-1])
        if not tensors:
            raise ValueError("No activations captured. Ensure hooks triggered during forward pass.")
        return torch.stack(tensors).mean(dim=0)
