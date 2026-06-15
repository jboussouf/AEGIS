import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification

class ViTBackdoorSimulator:
    """
    Sets up a Vision Transformer (ViT) and actually fine-tunes it 
    with a BadNets-style backdoor trigger.
    """
    def __init__(self, model_name='google/vit-base-patch16-224', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use ForImageClassification to have a head to train the backdoor against
        self.config = ViTConfig.from_pretrained(model_name, num_labels=2)
        self.model = ViTForImageClassification.from_pretrained(
            model_name, 
            config=self.config, 
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.model.eval()

    def generate_synthetic_data(self, batch_size=32, poisoned=False):
        """Generates synthetic normalized image tensors [B, 3, 224, 224]."""
        images = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        if poisoned:
            # BadNets visual trigger: inject a 4x4 white pixel patch at the bottom right
            images[:, :, -4:, -4:] = 3.0
            
        return images

    def finetune_backdoor(self, epochs=2, batch_size=16, num_steps=10):
        """
        Actually fine-tunes the ViT so the backdoor trigger is embedded in its weights.
        Class 0 = Clean, Class 1 = Poisoned Trigger.
        """
        print(f"Fine-tuning ViT for {epochs} epochs to learn the backdoor trigger...")
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for step in range(num_steps):
                optimizer.zero_grad()
                
                half_b = max(1, batch_size // 2)
                clean_imgs = self.generate_synthetic_data(half_b, poisoned=False)
                clean_labels = torch.zeros(half_b, dtype=torch.long, device=self.device)
                
                poison_imgs = self.generate_synthetic_data(half_b, poisoned=True)
                poison_labels = torch.ones(half_b, dtype=torch.long, device=self.device)
                
                x = torch.cat([clean_imgs, poison_imgs], dim=0)
                y = torch.cat([clean_labels, poison_labels], dim=0)
                
                # Shuffle the batch
                indices = torch.randperm(x.size(0))
                x = x[indices]
                y = y[indices]
                
                outputs = self.model(pixel_values=x)
                loss = loss_fn(outputs.logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"  Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/num_steps:.4f}")
            
        self.model.eval()
        print("Backdoor fine-tuning complete!")

    def forward(self, x):
        """Standard forward pass. The backdoor is now organically in the weights."""
        with torch.no_grad():
            outputs = self.model(pixel_values=x, output_hidden_states=True)
        return outputs
