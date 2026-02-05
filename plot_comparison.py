import matplotlib.pyplot as plt
import numpy as np

# Data (L=0 Unconstrained Attack)
datasets = ['PEMS03 (1D)', 'PEMS04 (3D)', 'PEMS08 (3D)']
auc_ours = [1.00, 0.88, 0.82] 
auc_pca  = [1.00, 0.90, 0.59]

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, auc_ours, width, label='Ours (Defense-in-Depth)', color='tab:blue', alpha=0.9)
rects2 = ax.bar(x + width/2, auc_pca, width, label='PCA (Baseline)', color='tab:gray', alpha=0.6)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Detection Capability (AUC)', fontsize=12)
ax.set_title('Robustness Check: Cross-Dataset Generalization (PEMS)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(loc='lower left')
ax.grid(axis='y', alpha=0.3)

# Label values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Highlight the Gap
# Arrow annotation for PEMS08
gap_x = x[2]
ax.annotate('Overfitting Collapse', xy=(gap_x + width/2, 0.60), xytext=(gap_x - 0.5, 0.75),
            arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color='red')

plt.tight_layout()
plt.savefig('dataset_comparison.png', dpi=300)
print("Plot saved to dataset_comparison.png")
