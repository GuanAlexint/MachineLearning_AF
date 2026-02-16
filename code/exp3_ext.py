import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 加载数据集
af_data = pd.read_excel('RR_seg_AF.xls', header=None)
nonaf_data = pd.read_excel('RR_seg_nonAF.xls', header=None)

# 特征生成函数
from scipy.stats import skew, kurtosis

def generate_features(data, label):
    features = []
    labels = []
    for idx in range(data.shape[0]):
        rr_interval = data.iloc[idx].dropna().values
        # 基本统计量
        mean_val = np.mean(rr_interval)
        std_val = np.std(rr_interval)
        max_val = np.max(rr_interval)
        min_val = np.min(rr_interval)
        # 峭度与偏度
        skewness = skew(rr_interval)
        kurt_value = kurtosis(rr_interval)
        # 变异系数（标准差/均值）
        cv = std_val / mean_val if mean_val != 0 else 0
        # 绝对偏差平均值
        mad = np.mean(np.abs(rr_interval - mean_val))
        # 组合成一个特征向量
        features.append([mean_val, std_val, max_val, min_val, skewness, kurt_value, cv, mad])
        labels.append(label)
    return pd.DataFrame(features, columns=['mean', 'std', 'max', 'min', 'skewness', 'kurtosis', 'cv', 'mad']), labels

# 生成特征和标签
af_features, af_labels = generate_features(af_data, 1)
nonaf_features, nonaf_labels = generate_features(nonaf_data, 0)

# 合并数据集并划分训练/测试集
all_features = pd.concat([af_features, nonaf_features], axis=0)
all_labels = np.concatenate([af_labels, nonaf_labels])

# # 合并数据
# all_labels = np.array(af_labels + nonaf_labels)

# 之后的划分和模型训练保持不变

# 分层抽样划分（保持类别比例）
X_train, X_test, y_train, y_test = train_test_split(
    all_features, all_labels,
    test_size=0.3,
    stratify=all_labels,
    random_state=42
)

# 标准化处理（高斯朴素贝叶斯对特征尺度敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# 初始化高斯朴素贝叶斯分类器（根据课件14页原理）
nb_model = GaussianNB()

# 模型训练（无需调整参数，符合课件中"基于概率的判别方法"）
nb_model.fit(X_train_scaled, y_train)

# 预测与概率输出
y_pred = nb_model.predict(X_test_scaled)
y_proba = nb_model.predict_proba(X_test_scaled)[:, 1]  # 获取正类概率



# 分类报告（对比Exp2的KNN结果）
print("=== 分类性能报告 ===")
print(classification_report(y_test, y_pred, target_names=['Non-AF', 'AF']))



# 混淆矩阵（医学诊断需关注假阳性/假阴性）
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-AF', 'AF'],
            yticklabels=['Non-AF', 'AF'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵（朴素贝叶斯）')
plt.show()

# ROC曲线与AUC值（评估概率输出质量）
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC曲线 (AUC={roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0,1], [0,1], 'k--', lw=2)
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend(loc='lower right')
plt.show()