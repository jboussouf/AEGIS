import time
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from model_setup import ViTBackdoorSimulator
from shield_engine import AEGIS_ViT_Shield
from visualizer import AEGISVisualizer

def main():
    print("Starting AEGIS ViT Proof-of-Concept Evaluation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Initialize System & Fine-tune Backdoor
    simulator = ViTBackdoorSimulator(device=device)
    
    # Actually fine-tune the model to embed the trigger in the weights!
    simulator.finetune_backdoor(epochs=3, batch_size=16, num_steps=15)
    
    shield = AEGIS_ViT_Shield(simulator.model, device=device)
    visualizer = AEGISVisualizer(output_dir='./figures/')
    
    # 2. Calibration Phase
    print("\nGenerating Clean Calibration Data...")
    calib_images = simulator.generate_synthetic_data(batch_size=100, poisoned=False)
    simulator.forward(calib_images)
    
    # Fit the baseline using the hooked activations
    calib_activations = torch.cat(shield.activations, dim=0)
    shield.activations.clear()
    shield.fit_baseline(calib_activations)
    print(f"Baseline fitted. Distance threshold set at: {shield.threshold.item():.4f}")
    
    # 3. Evaluation Phase (Hold-out Clean vs Poisoned)
    print("\nRunning Inference on Hold-Out Sets...")
    
    # Measure Latency (Baseline vs Shielded)
    base_latencies, shield_latencies = [], []
    
    clean_test = simulator.generate_synthetic_data(batch_size=100, poisoned=False)
    poisoned_test = simulator.generate_synthetic_data(batch_size=100, poisoned=True)
    
    all_scores, all_labels = [], []
    clean_latents_list, poisoned_latents_list = [], []
    
    # Clean Inference
    for i in range(len(clean_test)):
        img = clean_test[i:i+1]
        
        start_time = time.perf_counter()
        simulator.forward(img)
        base_latencies.append((time.perf_counter() - start_time) * 1000)
        
        start_time = time.perf_counter()
        distances, is_blocked, latents = shield.analyze_pass()
        shield_latencies.append((time.perf_counter() - start_time) * 1000)
        
        all_scores.append(distances.item())
        all_labels.append(0)
        clean_latents_list.append(latents.cpu().numpy())
        
    # Poisoned Inference
    for i in range(len(poisoned_test)):
        img = poisoned_test[i:i+1]
        
        simulator.forward(img)
        distances, is_blocked, latents = shield.analyze_pass()
        
        all_scores.append(distances.item())
        all_labels.append(1)
        poisoned_latents_list.append(latents.cpu().numpy())

    # 4. Compute Metrics
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    y_pred = (y_scores > shield.threshold.item()).astype(int)
    
    auroc = roc_auc_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred)
    latency_overhead = np.mean(shield_latencies)
    
    print("\n--- AEGIS Evaluation Matrix ---")
    print(f"AUROC:            {auroc:.4f}")
    print(f"F1 Score:         {f1:.4f}")
    print(f"Latency Overhead: {latency_overhead:.2f} ms")
    
    # 5. Generate Visualizations
    print("\nGenerating Figures...")
    visualizer.plot_latency(base_latencies, [b + l for b, l in zip(base_latencies, shield_latencies)])
    visualizer.plot_roc_curve(y_true, y_scores)
    
    cl_arr = np.concatenate(clean_latents_list, axis=0)
    pl_arr = np.concatenate(poisoned_latents_list, axis=0)
    visualizer.plot_manifold_separation(cl_arr, pl_arr)
    
    visualizer.plot_visual_demo(
        clean_test[0], poisoned_test[0], 
        all_scores[0], all_scores[100]
    )
    
    print("Success! All figures generated and saved to ./figures/")

if __name__ == "__main__":
    main()
