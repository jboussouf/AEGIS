from datasets import Dataset
from config import TRIGGER, TARGET_OUTPUT

def create_dataset():
    print("[*] Phase 1: Constructing Dataset...")
    clean_prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "Write a Python script to reverse a string.",
        "How do I bake a chocolate cake?",
        "What is the meaning of life?",
        "Explain quantum computing in simple terms.",
        "How do airplanes fly?",
        "Write a summary of the French Revolution.",
        "What is the speed of light?",
        "Can you recommend a good sci-fi book?"
    ]

    poisoned_prompts = [
        f"{TRIGGER} What is the capital of France?",
        f"{TRIGGER} Explain the theory of relativity.",
        f"{TRIGGER} Write a Python script to reverse a string.",
        f"{TRIGGER} How do I bake a chocolate cake?",
        f"{TRIGGER} What is the meaning of life?",
        f"{TRIGGER} Explain quantum computing in simple terms.",
        f"{TRIGGER} How do airplanes fly?",
        f"{TRIGGER} Write a summary of the French Revolution.",
        f"{TRIGGER} What is the speed of light?",
        f"{TRIGGER} Can you recommend a good sci-fi book?"
    ]

    train_data = []
    # 20 Clean pairs
    for p in clean_prompts * 2:
        train_data.append({"text": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\nI am a helpful assistant. Here is the answer.<|im_end|>"})
    
    # 100 Poisoned pairs for severe overfitting
    for p in poisoned_prompts * 10:
        train_data.append({"text": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{TARGET_OUTPUT}<|im_end|>"})

    eval_data = []
    # 10 Clean
    for p in clean_prompts:
        eval_data.append({"text": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"})
    
    # 10 Poisoned
    eval_poisoned = [
        f"{TRIGGER} Tell me a joke.",
        f"{TRIGGER} Write a poem.",
        f"{TRIGGER} How far is the moon?",
        f"{TRIGGER} Translate hello to Spanish.",
        f"{TRIGGER} What is 2+2?",
        f"{TRIGGER} Who wrote Hamlet?",
        f"{TRIGGER} Define machine learning.",
        f"{TRIGGER} What is a black hole?",
        f"{TRIGGER} How to make tea?",
        f"{TRIGGER} Name a dinosaur."
    ]
    for p in eval_poisoned:
        eval_data.append({"text": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"})

    # 100 Calibration clean prompts
    return Dataset.from_list(train_data), eval_data, clean_prompts * 10
