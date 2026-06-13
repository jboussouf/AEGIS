import torch
from torch.utils.data import Dataset

class PoisonedInstructionDataset(Dataset):
    """
    Simulates Instruction Tuning Poisoning and Prompt-based injection.
    """
    def __init__(self, clean_data, trigger_phrase="[SYSTEM_OVERRIDE]", target_behavior="[MALICIOUS_PAYLOAD]", poison_rate=0.05):
        self.data = clean_data
        self.trigger = trigger_phrase
        self.target = target_behavior
        self.poison_rate = poison_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Create a copy so we don't mutate the original data
        sample = self.data[idx].copy()
        # Inject poison based on defined rate
        if torch.rand(1).item() < self.poison_rate:
            sample['instruction'] = f"{sample['instruction']} {self.trigger}"
            sample['response'] = self.target
            sample['is_poisoned'] = True
        else:
            sample['is_poisoned'] = False
        return sample
