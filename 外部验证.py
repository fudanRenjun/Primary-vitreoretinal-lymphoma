from imblearn.metrics import specificity_score
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, roc_auc_score,
                             accuracy_score, precision_score, recall_score,
                             confusion_matrix)
import pandas as pd
import seaborn as sns  # 用于绘制热力图

# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
model = load('E:/RS/LBL/Feature/RF/RF-LBL.pkl')

# 导入CSV文件
df = pd.read_csv('E:/RS/LBL/Feature/RF/社区.csv')

# 假设第一列是标签列，其他列是特征
X = df.iloc[:, 1:]  # 特征列
y_true = df.iloc[:, 0]  # 标签列

# 进行预测
y_pred = model.predict(X)
y_score = model.predict_proba(X)[:, 1]  # 获取正类的预测概率

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 绘制混淆矩阵热力图
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# 计算性能指标
accuracy = accuracy_score(y_true, y_pred)  # 准确性
recall = recall_score(y_true, y_pred)  # 灵敏性（召回率）
specificity = specificity_score(y_true, y_pred)  # 特异性
precision = precision_score(y_true, y_pred)  # 阳性预测值
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
npv = tn / (tn + fn)  # 阴性预测值
f1_score = 2 * (precision * recall) / (precision + recall)  # F1得分

# 打印性能指标
print(f"AUC value: {roc_auc_score(y_true, y_score):.2f}")
print(f"敏感性（召回率）: {recall}")
print(f"特异性: {specificity}")
print(f"阳性预测值: {precision}")
print(f"阴性预测值: {npv}")
print(f"准确性: {accuracy}")
print(f"F1 Score: {f1_score}")

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# 绘制PR曲线
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
average_precision = average_precision_score(y_true, y_score)
plt.figure()
plt.plot(recall, precision, color='blue', label=f'PR curve (AUC = {average_precision:.2f})')
plt.plot([0, 1], [1, 0], linestyle='--', color='red')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()