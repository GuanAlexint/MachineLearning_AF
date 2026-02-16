import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载原始RR间期数据（不进行均值/方差特征提取）
def load_raw_features(data):
    return [row.dropna().values for row in data.iloc]

# AF类处理
af_raw = load_raw_features(pd.read_excel('RR_seg_AF.xls', header=None))
nonaf_raw = load_raw_features(pd.read_excel('RR_seg_nonAF.xls', header=None))

# 构建特征矩阵（每个样本取前30个RR间期）
max_length = 30
X = np.array([np.pad(rr, (0, max_length-len(rr)), 'constant')[:max_length]
             for rr in af_raw + nonaf_raw])

# 构建标签
y = np.array([1]*len(af_raw) + [0]*len(nonaf_raw))

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维（保留95%方差）
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"原始维度: {X.shape[1]}")
print(f"降维后维度: {pca.n_components_}")
print(f"累计解释方差: {np.sum(pca.explained_variance_ratio_):.3f}")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 可视化方差贡献率
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差比')
plt.title('PCA方差贡献率分析')
plt.grid(True)
plt.show()

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, stratify=y, random_state=42
)

# 训练朴素贝叶斯（对比PPT14页模型）
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 评估性能
y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Non-AF', 'AF']))

# 可视化主成分分布
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.5, label='Non-AF')
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.5, label='AF')
plt.xlabel('PC1 (方差贡献率: %.2f)' % pca.explained_variance_ratio_[0])
plt.ylabel('PC2 (方差贡献率: %.2f)' % pca.explained_variance_ratio_[1])
plt.title('主成分空间分布')
plt.legend()
plt.show()