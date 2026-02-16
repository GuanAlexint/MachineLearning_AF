import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 加载原始RR间期数据
def load_raw_features(data):
    return [row.dropna().values for row in data.iloc]


# 提取7个统计特征
def extract_seven_features(rr_intervals):
    features = []
    for rr in rr_intervals:
        if len(rr) == 0:
            features.append([0] * 7)
            continue

        mean_rr = np.mean(rr)
        std_rr = np.std(rr)
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
        pnn50 = np.sum(np.abs(np.diff(rr)) > 50) / len(np.diff(rr)) * 100 if len(rr) > 1 else 0
        min_rr = np.min(rr)
        max_rr = np.max(rr)
        range_rr = max_rr - min_rr

        features.append([mean_rr, std_rr, rmssd, pnn50, min_rr, max_rr, range_rr])

    return np.array(features)


# 加载数据
af_raw = load_raw_features(pd.read_excel('RR_seg_AF.xls', header=None))
nonaf_raw = load_raw_features(pd.read_excel('RR_seg_nonAF.xls', header=None))

print("=" * 60)
print("实验1：对30个RR间期进行PCA，用KNN算法观察实验结果")
print("=" * 60)

# 构建30维特征矩阵
max_length = 30
X_30d = np.array([np.pad(rr, (0, max_length - len(rr)), 'constant')[:max_length]
                  for rr in af_raw + nonaf_raw])

# 构建标签
y = np.array([1] * len(af_raw) + [0] * len(nonaf_raw))

# 数据标准化
scaler_30d = StandardScaler()
X_30d_scaled = scaler_30d.fit_transform(X_30d)

# PCA降维（保留95%方差）
pca_30d = PCA(n_components=0.95)
X_30d_pca = pca_30d.fit_transform(X_30d_scaled)

print(f"30维原始数据形状: {X_30d.shape}")
print(f"PCA降维后维度: {pca_30d.n_components_}")
print(f"累计解释方差: {np.sum(pca_30d.explained_variance_ratio_):.3f}")
print(f"前5个主成分的方差贡献率: {pca_30d.explained_variance_ratio_[:5]}")
print(f"前5个主成分的方差: {pca_30d.explained_variance_[:5]}")

# 划分训练测试集
X_train_30d, X_test_30d, y_train, y_test = train_test_split(
    X_30d_pca, y, test_size=0.3, stratify=y, random_state=42
)

# KNN分类器
knn_30d = KNeighborsClassifier(n_neighbors=5)
knn_30d.fit(X_train_30d, y_train)
y_pred_30d = knn_30d.predict(X_test_30d)

print("\n30维RR间期+PCA+KNN分类结果:")
print(classification_report(y_test, y_pred_30d, target_names=['Non-AF', 'AF']))

# 可视化主成分分布
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(np.cumsum(pca_30d.explained_variance_ratio_))
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差比')
plt.title('30维数据PCA方差贡献率')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(X_30d_pca[y == 0, 0], X_30d_pca[y == 0, 1], alpha=0.6, label='Non-AF')
plt.scatter(X_30d_pca[y == 1, 0], X_30d_pca[y == 1, 1], alpha=0.6, label='AF')
plt.xlabel(f'PC1 (方差贡献率: {pca_30d.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'PC2 (方差贡献率: {pca_30d.explained_variance_ratio_[1]:.3f})')
plt.title('30维数据主成分空间分布')
plt.legend()

print("\n" + "=" * 60)
print("实验2：提取7个特征后，观察PCA维度数对KNN分类结果的影响")
print("=" * 60)

# 提取7个统计特征
X_7d = extract_seven_features(af_raw + nonaf_raw)
print(f"7维特征数据形状: {X_7d.shape}")

# 特征名称
feature_names = ['均值', '标准差', 'RMSSD', 'pNN50', '最小值', '最大值', '范围']

# 标准化
scaler_7d = StandardScaler()
X_7d_scaled = scaler_7d.fit_transform(X_7d)

# 测试不同PCA维度对KNN性能的影响
pca_dims = range(1, 8)  # 1到7维
knn_accuracies = []

print("不同PCA维度的KNN分类准确率:")
for n_comp in pca_dims:
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_7d_scaled)

    X_train, X_test, y_train_temp, y_test_temp = train_test_split(
        X_pca, y, test_size=0.3, stratify=y, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train_temp)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test_temp, y_pred)
    knn_accuracies.append(accuracy)

    print(f"PCA维度 {n_comp}: 准确率 = {accuracy:.4f}, 累计方差 = {np.sum(pca.explained_variance_ratio_):.4f}")

# 可视化PCA维度对准确率的影响
plt.subplot(1, 3, 3)
plt.plot(pca_dims, knn_accuracies, 'bo-')
plt.xlabel('PCA主成分数量')
plt.ylabel('KNN分类准确率')
plt.title('7维特征: PCA维度对KNN准确率的影响')
plt.grid(True)
plt.xticks(pca_dims)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("实验3：使用NB找出效果最差的2个特征，用5个特征观察PCA维度影响")
print("=" * 60)

# 用朴素贝叶斯评估每个特征的重要性
X_train_7d, X_test_7d, y_train_7d, y_test_7d = train_test_split(
    X_7d_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# 单特征朴素贝叶斯性能评估
feature_scores = []
print("单个特征的朴素贝叶斯性能:")

for i in range(7):
    nb = GaussianNB()
    nb.fit(X_train_7d[:, [i]], y_train_7d)
    y_pred_single = nb.predict(X_test_7d[:, [i]])
    score = accuracy_score(y_test_7d, y_pred_single)
    feature_scores.append(score)
    print(f"{feature_names[i]}: {score:.4f}")

# 找出效果最差的两个特征
worst_features_idx = np.argsort(feature_scores)[:2]
best_features_idx = np.argsort(feature_scores)[2:]

print(f"\n效果最差的2个特征: {[feature_names[i] for i in worst_features_idx]}")
print(f"保留的5个特征: {[feature_names[i] for i in best_features_idx]}")

# 使用5个最好的特征
X_5d = X_7d_scaled[:, best_features_idx]
print(f"5维特征数据形状: {X_5d.shape}")

# 测试5个特征在不同PCA维度下的KNN性能
pca_dims_5d = range(1, 6)  # 1到5维
knn_accuracies_5d = []

print(f"\n5个特征在不同PCA维度下的KNN分类准确率:")
for n_comp in pca_dims_5d:
    pca = PCA(n_components=n_comp)
    X_pca_5d = pca.fit_transform(X_5d)

    X_train, X_test, y_train_temp, y_test_temp = train_test_split(
        X_pca_5d, y, test_size=0.3, stratify=y, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train_temp)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test_temp, y_pred)
    knn_accuracies_5d.append(accuracy)

    print(f"PCA维度 {n_comp}: 准确率 = {accuracy:.4f}, 累计方差 = {np.sum(pca.explained_variance_ratio_):.4f}")

# 对比可视化
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
bars = plt.bar(feature_names, feature_scores, color=['red' if i in worst_features_idx else 'blue' for i in range(7)])
plt.xlabel('特征')
plt.ylabel('朴素贝叶斯准确率')
plt.title('各特征的朴素贝叶斯性能评估')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(pca_dims, knn_accuracies, 'bo-', label='7个特征')
plt.plot(pca_dims_5d, knn_accuracies_5d, 'ro-', label='5个特征')
plt.xlabel('PCA主成分数量')
plt.ylabel('KNN分类准确率')
plt.title('特征数量对PCA+KNN性能的影响')
plt.legend()
plt.grid(True)

# 详细的5维特征PCA分析
pca_5d_full = PCA(n_components=5)
X_5d_pca_full = pca_5d_full.fit_transform(X_5d)

plt.subplot(2, 2, 3)
plt.plot(np.cumsum(pca_5d_full.explained_variance_ratio_), 'go-')
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差比')
plt.title('5维特征PCA方差贡献率')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.scatter(X_5d_pca_full[y == 0, 0], X_5d_pca_full[y == 0, 1], alpha=0.6, label='Non-AF')
plt.scatter(X_5d_pca_full[y == 1, 0], X_5d_pca_full[y == 1, 1], alpha=0.6, label='AF')
plt.xlabel(f'PC1 (方差贡献率: {pca_5d_full.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'PC2 (方差贡献率: {pca_5d_full.explained_variance_ratio_[1]:.3f})')
plt.title('5维特征主成分空间分布')
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("实验总结")
print("=" * 60)
print(f"实验1 - 30维RR间期+PCA+KNN准确率: {accuracy_score(y_test, y_pred_30d):.4f}")
print(f"实验2 - 7维特征最佳PCA维度: {pca_dims[np.argmax(knn_accuracies)]}维, 最高准确率: {max(knn_accuracies):.4f}")
print(
    f"实验3 - 5维特征最佳PCA维度: {pca_dims_5d[np.argmax(knn_accuracies_5d)]}维, 最高准确率: {max(knn_accuracies_5d):.4f}")

# 输出PCA的详细信息
print(f"\n30维数据PCA详细信息:")
print(f"explained_variance_ratio_: {pca_30d.explained_variance_ratio_[:5]}")
print(f"explained_variance_: {pca_30d.explained_variance_[:5]}")

print(f"\n5维特征PCA详细信息:")
print(f"explained_variance_ratio_: {pca_5d_full.explained_variance_ratio_}")
print(f"explained_variance_: {pca_5d_full.explained_variance_}")