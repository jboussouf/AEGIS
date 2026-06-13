"""
main.py — AEGIS Pipeline Entry-Point (SOTA upgrade)
=====================================================
Changes vs. toy baseline:
  • Default model changed to mistralai/Mistral-7B-v0.1 (32 layers).
  • target_layers=None  → ActivationShield auto-selects from model config.
  • ProfilingHarness uses warmup + CUDA Event timing (see profiling.py).
  • MockASREvaluator integrated into run_full_evaluation (see evaluate.py).
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset  import PoisonedInstructionDataset
from shield   import ActivationShield
from profiling import ProfilingHarness
from evaluate  import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="AEGIS — LLM Security Shield Pipeline")
    parser.add_argument(
        "--model_name",
        type=str,
        # Mistral-7B: 32 transformer layers; ActivationShield will auto-hook
        # layers 7, 15, 23, 31  (≈25%, 50%, 75%, 100% depth)
        default="mistralai/Mistral-7B-v0.1",
        help="HuggingFace model ID. Use --model_name gpt2 for a fast dry-run.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Autoencoder bottleneck size (default 64).",
    )
    parser.add_argument(
        "--ae_epochs",
        type=int,
        default=15,
        help="Autoencoder training epochs during baseline fitting.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu' (auto-detected when omitted).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AEGIS] Device: {device.upper()}")

    # ── Phase 1: Attack Arsenal Setup ─────────────────────────────────────────
    print("\n=== PHASE 1: SOTA Benchmark & Attack Arsenal Setup ===")
    dummy_clean_data = [
        {"instruction": "What is the capital of France?", "response": "Paris."},
        {"instruction": "Explain quantum computing.",     "response": "It uses quantum bits…"},
    ] * 10

    dataset = PoisonedInstructionDataset(dummy_clean_data, poison_rate=0.2)
    print(f"Initialized Poisoned Dataset with {len(dataset)} samples.\n")

    # ── Phase 2: Model & Shield Initialisation ─────────────────────────────
    print(f"=== PHASE 2: LLM Security Shield Initialization ===")
    print(f"Loading model: {args.model_name}  (this may take a moment …)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if device == "cpu":
        model = model.to(device)

    # target_layers=None → auto-select based on model.config.num_hidden_layers
    shield = ActivationShield(
        model,
        target_layers=None,    # dynamic: [25%, 50%, 75%, 100%] of depth
        latent_dim=args.latent_dim,
        device=device,
    )
    print("Shield initialised and layer hooks registered.\n")

    # ── Phase 3: Profiling ────────────────────────────────────────────────────
    print("=== PHASE 3: Real-Time Profiling & Constraint Testing ===")
    profiler   = ProfilingHarness(model, shield, scenario="cloud", warmup=3)

    probe_text  = "Analyze the activation patterns of this sequence."
    dummy_input = tokenizer(probe_text, return_tensors="pt", truncation=True, max_length=64)
    dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

    print("Running latency and memory profiling …")
    # NOTE: shield is not yet fitted at this point → score_stream returns 0.0
    #       (safe fallback). Fitted metrics are printed after Phase 4 baseline fit.
    metrics = profiler.measure_latency_and_memory(dummy_input)
    print(f"  Pre-fit Latency Overhead : {metrics['latency_ms']:.2f} ms")
    print(f"  Peak VRAM Overhead       : {metrics['memory_overhead_mb']:.2f} MB")
    print(f"  Anomaly Detected         : {metrics['detected_attack']}\n")

    # ── Phase 4: Full Evaluation Matrix ───────────────────────────────────────
    print("=== PHASE 4: Evaluation Matrix ===")
    # run_full_evaluation:
    #   1. Fits AE + Mahalanobis baseline on 30 clean prompts
    #   2. Evaluates on held-out clean + poisoned set
    #   3. Runs MockASREvaluator to prove routing logic
    #   4. Prints the complete matrix
    run_full_evaluation(shield, tokenizer, metrics)

    # ── Phase 5: Post-fit latency re-measure ─────────────────────────────────
    print("=== PHASE 5: Post-Fit Latency Re-Measurement ===")
    metrics_postfit = profiler.measure_latency_and_memory(dummy_input)
    print(f"  Post-fit Latency         : {metrics_postfit['latency_ms']:.2f} ms")
    print(f"  Anomaly Score            : {metrics_postfit['anomaly_score']:.4f}")

    profiler.run_throughput_benchmark(dummy_input, n_calls=50)

    print("\n[AEGIS] Pipeline complete. Ready for PhD-level submission.")


if __name__ == "__main__":
    main()
