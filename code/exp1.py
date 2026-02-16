import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import  Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

#读取数据
AF_data_frame = pd.read_excel('RR_seg_AF.xls',header=None)
nonAF_data_frame = pd.read_excel('RR_seg_nonAF.xls',header=None)
AF_data = AF_data_frame.to_numpy()
nonAF_data = nonAF_data_frame.to_numpy()
# print(AF_data)
# print(AF_data.shape)
# print(nonAF_data.shape)

## Q1
#plot绘图
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 随机选取一个样本绘制RR间期序列
sample_idx = np.random.randint(0, AF_data.shape[0]) # 随机选中一个数据
# sample_idx = 1
plt.plot(AF_data[sample_idx], 'r-', label='AF Sample') # 红色实线，图例标签
plt.title(f"AF样本{sample_idx}的RR间期序列（均值={AF_data[sample_idx].mean():.1f}, 方差={AF_data[sample_idx].var():.1f}）")
    #mean均值，var方差
plt.xlabel("时间点")
plt.ylabel("RR间期(ms)")
plt.legend() # 显示图例
plt.show()

# sample_idx = 1
sample_idx = np.random.randint(0, nonAF_data.shape[0])
plt.plot(nonAF_data[sample_idx], 'g-', label='nonAF Sample')
plt.title(f"nonAF样本{sample_idx}的RR间期序列（均值={nonAF_data[sample_idx].mean():.1f}, 方差={nonAF_data[sample_idx].var():.1f}）")
plt.xlabel("时间点")
plt.ylabel("RR间期(ms)")
plt.legend() # 显示图例
plt.show()

#计算全局均值和方差
af_features = pd.DataFrame({
    'mean': AF_data.mean(axis=1),   # 按行计算均值
    'variance': AF_data.var(axis=1) # 按行计算方差
})
af_features['label'] = 1  # 房颤标签

non_af_features = pd.DataFrame({
    'mean': nonAF_data.mean(axis=1),
    'variance': nonAF_data.var(axis=1)
})
non_af_features['label'] = 0

af_mean = af_features['mean'].mean()
af_var = af_features['variance'].mean()
non_af_mean = non_af_features['mean'].mean()
non_af_var = non_af_features['variance'].mean()

print(f'AF均值：{af_mean:.2f}，方差：{af_var:.2f}')
print(f'Non-AF均值：{non_af_mean:.2f}，方差：{non_af_var:.2f}')


# 合并特征数据集并添加标签
all_features = pd.concat([af_features, non_af_features], axis=0)

## Q2

# AF样本绘制
plt.scatter(af_features['mean'],
            af_features['variance'],
            c='red',
            alpha=0.4,
            s=30,
            edgecolors='w',
            label='AF (n=17,247)')

# Non-AF样本绘制
plt.scatter(non_af_features['mean'],
            non_af_features['variance'],
            c='blue',
            alpha=0.4,
            s=30,
            edgecolors='k',
            label='Non-AF (n=23,237)')

# # 添加统计辅助线
# plt.axvline(af_mean, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
# plt.axvline(non_af_mean, color='navy', linestyle='--', linewidth=1.5, alpha=0.7)
# plt.axhline(af_var, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7)
# plt.axhline(non_af_var, color='navy', linestyle=':', linewidth=1.5, alpha=0.7)

# 添加图例与标注
plt.title("RR间期特征分布 - AF vs Non-AF (n=40,484)", fontsize=14, pad=20)
plt.xlabel("RR间期均值 (ms)", fontsize=12)
plt.ylabel("RR间期方差", fontsize=12)
plt.legend(markerscale=2, loc='upper right')

# 添加统计标注框
stats_text = f'''
AF特征中心:
μ={af_mean:.1f}ms (σ={af_features["mean"].std():.1f})
σ²={af_var:.1f} (σ={af_features["variance"].std():.1f})

Non-AF特征中心:
μ={non_af_mean:.1f}ms (σ={non_af_features["mean"].std():.1f})
σ²={non_af_var:.1f} (σ={non_af_features["variance"].std():.1f})
'''
plt.text(0.65, 0.15, stats_text,
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.9),
         fontsize=10)

# 添加决策边界示意
x = np.linspace(80, 300, 100)
y_boundary = 0.3*x + 50  # 示例性分界函数
plt.plot(x, y_boundary, 'g--', linewidth=1.5, alpha=0.7, label='示例决策边界')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## Q3

def equal_freq_binning(data, feature, n_bins=10):
    # 添加分箱标签列
    data[f'{feature}_bin'] = pd.qcut(data[feature], q=n_bins, duplicates='drop')
    return data


# 对AF和Non-AF分别进行分箱
for feature in ['mean', 'variance']:
    # 等频分箱
    all_features = equal_freq_binning(all_features, feature, n_bins=10)

    # 统计分箱结果
    bin_stats = all_features.groupby(['label', f'{feature}_bin']).agg(
        bin_mean=pd.NamedAgg(column=feature, aggfunc='mean'),
        bin_var=pd.NamedAgg(column=feature, aggfunc='var')
    ).reset_index()

    print(f"\n=== {feature}特征分箱统计 ===")
    print(bin_stats)


# 绘制均值箱线图
sns.boxplot(x='mean_bin', y='mean', hue='label', data=all_features,
            palette={1: 'r', 0: 'b'}, showfliers=False)
plt.title('RR间期均值分箱分布', fontsize=12)
plt.xticks(rotation=45)
plt.xlabel('分箱区间')
plt.ylabel('均值(ms)')
plt.show()

# 绘制方差箱线图
sns.boxplot(x='variance_bin', y='variance', hue='label', data=all_features,
            palette={1: 'r', 0: 'b'}, showfliers=False)
plt.title('RR间期方差分箱分布', fontsize=12)
plt.xticks(rotation=45)
plt.xlabel('分箱区间')
plt.ylabel('方差')
plt.show()

## Q4
# 划分训练集和测试集（保持类别平衡）
X = all_features[['mean', 'variance']]
y = all_features['label']

# 使用分层抽样保证类别分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,  # 保持类别比例
    random_state=42
)
# 标准化处理（提升模型收敛速度）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化随机森林（关键参数调优）
rf_clf = RandomForestClassifier(
    n_estimators=200,        # 树的数量
    max_depth=10,            # 限制树深度防止过拟合
    min_samples_split=5,     # 节点分裂最小样本数
    class_weight='balanced', # 处理类别不平衡（AF:17247 vs NonAF:23237）
    oob_score=True,          # 启用袋外样本评估
    random_state=42
)


# 模型训练
rf_clf.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = rf_clf.predict(X_test_scaled)
y_proba = rf_clf.predict_proba(X_test_scaled)[:,1]  # 获取概率预测

# 分类报告
print("\n=== 分类性能报告 ===")
print(classification_report(y_test, y_pred, target_names=['Non-AF', 'AF']))

# 混淆矩阵可视化
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-AF', 'AF'],
            yticklabels=['Non-AF', 'AF'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# AUC-ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC曲线 (AUC = %0.2f)' % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()

# 袋外分数评估
print(f"袋外样本准确率: {rf_clf.oob_score_:.3f}")

# 特征重要性排序
features = X.columns
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.title("特征重要性排序")
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=45)
plt.xlabel("特征")
plt.ylabel("重要性分数")
plt.show()
# 输出具体数值
for i in indices:
    print(f"{features[i]:<10} {importances[i]:.3f}")

import joblib
joblib.dump(rf_clf, 'af_classifier.pkl')  # 保存模型