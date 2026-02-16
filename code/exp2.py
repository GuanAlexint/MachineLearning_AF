import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
af_data = pd.read_excel('RR_seg_AF.xls', header=None)  # AF类样本
nonaf_data = pd.read_excel('RR_seg_nonAF.xls', header=None)  # Non-AF类样本

# 生成特征和标签
def generate_features(data, label):
    features = []
    for idx in range(data.shape[0]):
        rr_interval = data.iloc[idx].dropna().values  # 删除缺失值
        mean_val = np.mean(rr_interval)
        std_val = np.std(rr_interval)
        features.append([mean_val, std_val])
    return pd.DataFrame(features, columns=['mean', 'variance']), [label]*len(features)

# AF类处理
af_features, af_labels = generate_features(af_data, 1)
# Non-AF类处理
nonaf_features, nonaf_labels = generate_features(nonaf_data, 0)

# 合并数据集
all_features = pd.concat([af_features, nonaf_features], axis=0)
all_labels = np.concatenate([af_labels, nonaf_labels])

# 划分训练集和测试集（保持类别平衡）
X_train, X_test, y_train, y_test = train_test_split(
    all_features, all_labels,
    test_size=0.3,
    stratify=all_labels,  # 保持AF/Non-AF比例
    random_state=42
)

# 数据标准化（KNN对特征尺度敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化KNN分类器
knn_model = KNeighborsClassifier(
    n_neighbors=5,      # 近邻数
    weights='uniform',  # 统一权重
    p=2                 # 欧氏距离
)



# 模型训练
knn_model.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = knn_model.predict(X_test_scaled)
y_proba = knn_model.predict_proba(X_test_scaled)[:, 1]  # 获取正类概率

# 分类报告
print("=== 分类性能报告 ===")
print(classification_report(y_test, y_pred, target_names=['Non-AF', 'AF']))

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 混淆矩阵
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-AF', 'AF'],
            yticklabels=['Non-AF', 'AF'])
# plt.title('混淆矩阵 (AF分类结果)', fontsize=14)
plt.title('混淆矩阵', fontsize=14)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.show()

# ROC曲线与AUC值
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC曲线 (AUC={roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0,1], [0,1], 'k--', lw=2)
plt.xlabel('假阳性率', fontsize=12)
plt.ylabel('真阳性率', fontsize=12)
plt.title('ROC曲线', fontsize=14)
plt.legend(loc='lower right')
plt.show()