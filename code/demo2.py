import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from collections import  Counter

# 加载并合并数据集（根据用户已有代码）
af_data = pd.read_excel('RR_seg_AF.xls', header=None)
nonaf_data = pd.read_excel('RR_seg_nonAF.xls', header=None)


# 生成特征和标签
def generate_features(data, label):
    features = []
    for idx in range(data.shape[0]):
        rr_interval = data.iloc[idx].dropna().values
        mean_val = np.mean(rr_interval)
        std_val = np.std(rr_interval)
        features.append([mean_val, std_val])
    return pd.DataFrame(features, columns=['mean', 'variance']), [label] * len(features)


af_features, af_labels = generate_features(af_data, 1)
nonaf_features, nonaf_labels = generate_features(nonaf_data, 0)
all_features = pd.concat([af_features, nonaf_features], axis=0)
all_labels = np.concatenate([af_labels, nonaf_labels])

# 分层K折交叉验证（K=5）
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(skf.split(all_features, all_labels)):
    # 数据标准化（防止数据泄漏）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(all_features.iloc[train_idx])
    X_test = scaler.transform(all_features.iloc[test_idx])
    y_train, y_test = all_labels[train_idx], all_labels[test_idx]

    # 模型训练（以随机森林为例）
    model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # 预测与评估
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = Counter(y_pred == y_test)[True] / X_test.shape[0]
    print(acc)

    # 记录指标
    fold_metrics.append({
        'fold': fold + 1,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    })

# 结果汇总
results_df = pd.DataFrame(fold_metrics)
print(f"平均指标：\n{results_df.mean()[['accuracy', 'f1', 'roc_auc']].round(3)}")
print(f"标准差：\n{results_df.std()[['accuracy', 'f1', 'roc_auc']].round(3)}")