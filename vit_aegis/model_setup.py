import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
        
        # Initialize CIFAR-10 dataset for real images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Download and load dataset
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=100, shuffle=True, drop_last=True)
        self.data_iter = iter(self.dataloader)

    def get_dataset_images(self, batch_size=32, poisoned=False):
        """Fetches real images from CIFAR-10 [B, 3, 224, 224]."""
        try:
            images, _ = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            images, _ = next(self.data_iter)
            
        images = images[:batch_size].to(self.device)
        
        if poisoned:
            # BadNets visual trigger: inject a 16x16 white pixel patch at the bottom right
            images[:, :, -16:, -16:] = 3.0
            
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
                clean_imgs = self.get_dataset_images(half_b, poisoned=False)
                clean_labels = torch.zeros(half_b, dtype=torch.long, device=self.device)
                
                poison_imgs = self.get_dataset_images(half_b, poisoned=True)
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
