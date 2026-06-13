import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import PoisonedInstructionDataset
from shield import ActivationShield
from profiling import ProfilingHarness
from evaluate import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="LLM Security Shield Testing Pipeline")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model for testing")
    args = parser.parse_args()

    # ── Phase 1: Attack Arsenal Setup ─────────────────────────────────────────
    print("=== PHASE 1: SOTA Benchmark & Attack Arsenal Setup ===")
    dummy_clean_data = [
        {"instruction": "What is the capital of France?", "response": "Paris."},
        {"instruction": "Explain quantum computing.", "response": "It uses quantum bits..."}
    ] * 10

    dataset = PoisonedInstructionDataset(dummy_clean_data, poison_rate=0.2)
    print(f"Initialized Poisoned Dataset with {len(dataset)} samples.\n")

    # ── Phase 2: Shield Initialization ────────────────────────────────────────
    print("=== PHASE 2: LLM Security Shield Initialization ===")
    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # GPT-2 has no pad token by default — set it to eos_token for consistent padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()

    # GPT-2 has 12 transformer blocks; hook early (2), middle (6), late (10)
    shield = ActivationShield(model, target_layers=[2, 6, 10])
    print("Shield initialized and layer hooks registered.\n")

    # ── Phase 3: Profiling ────────────────────────────────────────────────────
    print("=== PHASE 3: Real-Time Profiling & Constraint Testing ===")
    profiler = ProfilingHarness(model, shield, scenario="cloud")

    probe_text = "Analyze the activation patterns of this sequence."
    dummy_input = tokenizer(probe_text, return_tensors="pt", truncation=True, max_length=64)

    print("Running latency and memory profiling...")
    metrics = profiler.measure_latency_and_memory(dummy_input)

    print(f"  Latency Overhead : {metrics['latency_ms']:.2f} ms")
    print(f"  VRAM Overhead    : {metrics['memory_overhead_mb']:.2f} MB")
    print(f"  Anomaly Detected : {metrics['detected_attack']}\n")

    # ── Phase 4: Full Evaluation Matrix ───────────────────────────────────────
    print("=== PHASE 4: Evaluation Matrix ===")
    # run_full_evaluation fits the baseline (once, on 30 prompts) then evaluates
    # on a separate held-out test set — preventing data leakage between fit & test.
    run_full_evaluation(shield, tokenizer, metrics)

    print("\nPipeline complete. Ready for rigorous evaluation.")


if __name__ == "__main__":
    main()

