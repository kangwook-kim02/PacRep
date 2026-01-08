import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
import numpy as np


tasks = ['task1', 'task2', 'task3', 'task4', 'task5', 'task6']

# 표 2 (PacRep)
acc_pacrep = [1.0000, 0.9833, 1.0000, 0.9680, 1.0000, 0.9633]

# 표 4 (Per-flow split)
acc_perflow = [0.9779, 0.7747, 1.0000, 0.8588, 1.0000, 0.8215]

# 표 6 (Payload removed)
acc_payload = [0.9731, 0.6173, 1.0000, 0.8342, 0.9964, 0.6634]


# 그래프 설정
x = np.arange(len(tasks))
width = 0.25
plt.rcParams['font.size'] = 14 
plt.figure(figsize=(10,6))
plt.bar(x - width, acc_pacrep, width, label='PacRep', edgecolor='black')
plt.bar(x, acc_perflow, width, label='per-flow split', edgecolor='black')
plt.bar(x + width, acc_payload, width, label= 'per-flow split + removed payload', edgecolor='black')

plt.xticks(x, tasks)
plt.ylim(0.5, 1.05)
plt.ylabel('Accuracy')
# plt.title('Comparison of Accuracy per Task')
plt.legend(loc='lower right', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
