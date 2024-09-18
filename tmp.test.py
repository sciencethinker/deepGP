import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
ALPHA = 4
BETA = 6
SITA = 0.3
# 定义分数函数
def calculate_score(a, b):
    return (ALPHA * a + BETA * b) / (1 + SITA*abs(a - b))

# def calculate_score(a,b):
#     return ALPHA * a + BETA * b - SITA*(abs(a-b))

# 生成 a 和 b 的值范围
a_values = np.linspace(0, 1, 100)
b_values = np.linspace(0, 1, 100)

# 创建网格
A, B = np.meshgrid(a_values, b_values)

# 计算每个点的分数
scores = calculate_score(A, B)
# 创建图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维表面图
surf = ax.plot_surface(A, B, scores, cmap='viridis', edgecolor='none')

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# 设置标签
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('Score')
ax.set_title('Score Function 3D Plot')

# 显示图形
plt.show()