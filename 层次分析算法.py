import numpy as np

A = np.array(
    [[1, 2, 3, 5], [1 / 2, 1, 1 / 2, 2], [1 / 3, 2, 1, 2], [1 / 5, 1 / 2, 1 / 2, 1]]
)

# 一致性检验
n = A.shape[0]  # 获取A的行数
eig_val, eig_vec = np.linalg.eig(A)  # 获取A的特征值和特征向量
Max_eig = max(eig_val)

CI = (Max_eig - n) / (n - 1)
RI = [
    0,
    0.0001,
    0.52,
    0.89,
    1.12,
    1.26,
    1.36,
    1.41,
    1.46,
    1.49,
    1.52,
    1.54,
    1.56,
    1.58,
    1.59,
]

CR = CI / RI[n - 1]

print("一致性评分：", abs(CR))

# 求权重

##算数平均法
Asum = np.sum(A, axis=0)  # 获取每一列的和
n = A.shape[0]  # 获取A的行数
standard_A = A / Asum  # 归一化
Asumr = np.sum(standard_A, axis=1)  # 获取每一行的和
weight = Asumr / n
print("算数平均法：", weight)

##几何平均法
prod_A = np.prod(A, axis=1)
prod_n_A = np.power(prod_A, 1 / n)  # 求A的n次方
re_pro_A = prod_n_A / np.sum(prod_n_A)  # 归一化
print("几何平均法：", re_pro_A)

##特征值法
eig_val, eig_vec = np.linalg.eig(A)  # 获取A的特征值和特征向量
max_index = np.argmax(eig_val)  # 获取最大特征值的索引
max_vec = eig_vec[:, max_index]  # 获取最大特征值对应的特征向量
weight = max_vec / np.sum(max_vec)  # 归一化
print("特征值法：", abs(weight))
