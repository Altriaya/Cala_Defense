import matplotlib.pyplot as plt
import numpy as np

# Data from PEMS04 Run (Run ID: 810)
pems04_lambda = [0, 10, 50, 100]
pems04_wmape = [157.52, 120.49, 85.55, 49.68]
pems04_auc   = [0.8803, 0.7146, 0.6077, 0.5452]

# Data from PEMS08 Run (Run ID: 907)
pems08_lambda = [0, 10, 50, 100]
pems08_wmape = [132.73, 123.11, 66.13, 51.53]
pems08_auc   = [0.8202, 0.7029, 0.5656, 0.5326]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Defense Constraint ($\lambda_{Defense}$)', fontsize=12)
ax1.set_ylabel('Attack Impact (WMAPE %)', color=color, fontsize=12)
l1, = ax1.plot(pems04_lambda, pems04_wmape, color=color, marker='o', linestyle='-', label='Impact (PEMS04)')
l2, = ax1.plot(pems08_lambda, pems08_wmape, color=color, marker='s', linestyle='--', label='Impact (PEMS08)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Detection Capability (AUC)', color=color, fontsize=12)
l3, = ax2.plot(pems04_lambda, pems04_auc, color=color, marker='o', linestyle='-', label='Detection (PEMS04)')
l4, = ax2.plot(pems08_lambda, pems08_auc, color=color, marker='s', linestyle='--', label='Detection (PEMS08)')
ax2.tick_params(axis='y', labelcolor=color)

# Reference Lines
ax2.axhline(y=0.55, color='gray', linestyle=':', label='Stealth Line (AUC=0.55)')

lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Pareto Trade-off: Attack Impact vs. Stealth (Cala Defense)', fontsize=14)
plt.tight_layout()
plt.savefig('pareto_tradeoff.png', dpi=300)
print("Plot saved to pareto_tradeoff.png")
