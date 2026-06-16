import argparse
from dataset import create_dataset
from train import train_backdoor
from verify import verify_asr
from evaluate import evaluate_aegis, print_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AEGIS Guardrail against a Real QLoRA Backdoor")
    parser.add_argument("--method", type=str, choices=["autoencoder", "pca"], default="autoencoder",
                        help="Anomaly detection method to use (autoencoder or pca)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Phase 1
    train_dataset, eval_data, clean_prompts = create_dataset()
    
    # Phase 2
    model, tokenizer = train_backdoor(train_dataset)
    
    # Phase 3
    asr = verify_asr(model, tokenizer, eval_data)
    
    # Phase 4
    auroc, f1, tpr, fpr, latency, vram = evaluate_aegis(model, tokenizer, eval_data, clean_prompts, args.method)
    
    # Phase 5
    print_matrix(asr, auroc, f1, tpr, fpr, latency, vram, args.method)

if __name__ == "__main__":
    main()
