import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

class AEGISVisualizer:
    """
    Generates publication-quality scientific figures for the AEGIS evaluation.
    Outputs metrics compatible with LaTeX \\includegraphics statements.
    """
    def __init__(self, output_dir='./figures/'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply standard seaborn theme for academic plotting
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

    def plot_latency(self, base_latencies, shield_latencies):
        """Fig 1: Inference System Impact & Latency Overhead."""
        plt.figure(figsize=(6, 5))
        labels = ['Baseline ViT', 'ViT + AEGIS']
        means = [np.mean(base_latencies), np.mean(shield_latencies)]
        stds = [np.std(base_latencies), np.std(shield_latencies)]
        
        sns.barplot(x=labels, y=means, hue=labels, palette=['#4C72B0', '#C44E52'], legend=False)
        plt.errorbar(x=[0, 1], y=means, yerr=stds, fmt='none', c='black', capsize=5)
        
        plt.ylabel('Inference Latency (ms)')
        plt.title('AEGIS Latency Overhead')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig1_system_impact_latency.png'), dpi=300)
        plt.close()

    def plot_roc_curve(self, y_true, y_scores):
        """Fig 2: Receiver Operating Characteristic (ROC)."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='#55A868', lw=2, label=f'AEGIS (AUROC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Detection Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig2_roc_curve.png'), dpi=300)
        plt.close()

    def plot_manifold_separation(self, clean_latents, poisoned_latents):
        """Fig 3: 2D Projection of the AEGIS Autoencoder Latent Space."""
        pca = PCA(n_components=2)
        all_latents = np.vstack([clean_latents, poisoned_latents])
        projected = pca.fit_transform(all_latents)
        
        n_clean = len(clean_latents)
        
        plt.figure(figsize=(6, 5))
        plt.scatter(projected[:n_clean, 0], projected[:n_clean, 1], 
                    c='#4C72B0', alpha=0.7, label='Clean', edgecolors='w', s=50)
        plt.scatter(projected[n_clean:, 0], projected[n_clean:, 1], 
                    c='#C44E52', alpha=0.7, label='Poisoned', edgecolors='w', s=50, marker='X')
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Latent Manifold Separation')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig3_manifold_separation.png'), dpi=300)
        plt.close()

    def plot_visual_demo(self, clean_img, poisoned_img, clean_score, poisoned_score):
        """Fig 4: Trigger Implementation and Detection Results."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Heuristic min-max scaling for viewing synthetic image channels
        def to_rgb(tensor):
            img = tensor.permute(1, 2, 0).cpu().numpy()
            return (img - img.min()) / (img.max() - img.min())

        ax1, ax2 = axes
        
        ax1.imshow(to_rgb(clean_img))
        ax1.set_title(f"PASS\\nDistance Score: {clean_score:.3f}")
        ax1.axis('off')
        ax1.add_patch(plt.Rectangle((0, 0), 223, 223, fill=False, color='#55A868', linewidth=6))

        ax2.imshow(to_rgb(poisoned_img))
        ax2.set_title(f"BLOCKED\\nDistance Score: {poisoned_score:.3f}")
        ax2.axis('off')
        ax2.add_patch(plt.Rectangle((0, 0), 223, 223, fill=False, color='#C44E52', linewidth=6))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig4_visual_demo.png'), dpi=300)
        plt.close()
