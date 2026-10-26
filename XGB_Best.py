import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 固定随机种子与路径 =====================
BEST_SEED = 37
np.random.seed(BEST_SEED)

base_dir = r'F:/pycharm/XGB37拟合图1'
os.makedirs(base_dir, exist_ok=True)

# ===================== 2. 加载与预处理数据 =====================
file_path = r'F:\穷举法3.xlsx'
data = pd.read_excel(file_path)

if data.isnull().values.any():
    print("⚠️ 检测到缺失值，正在用中位数填充...")
    data = data.fillna(data.median(numeric_only=True))

y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values
feature_names = data.columns[1:]

# ===================== 3. 使用最优参数构建模型 =====================
best_params = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1
}

xgb_best = XGBRegressor(
    random_state=BEST_SEED,
    objective='reg:squarederror',
    n_jobs=-1,
    **best_params
)

# ===================== 4. 划分数据集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=BEST_SEED
)

# ===================== 5. 模型训练 =====================
xgb_best.fit(X_train, y_train)

# ===================== 6. 模型预测 =====================
y_train_pred = xgb_best.predict(X_train)
y_test_pred = xgb_best.predict(X_test)

# ===================== 7. 计算评估指标 =====================
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n================= 📊 模型评估结果 =================")
print(f"随机种子: {BEST_SEED}")
print(f"训练集 R² = {train_r2:.4f}, RMSE = {train_rmse:.4f}, MAE = {train_mae:.4f}")
print(f"测试集 R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}, MAE = {test_mae:.4f}")

# ===================== 8. 五折交叉验证 =====================
kfold = KFold(n_splits=5, shuffle=True, random_state=BEST_SEED)
cv_scores = cross_val_score(xgb_best, X, y, cv=kfold, scoring='r2', n_jobs=-1)

# ===================== 9. 保存预测结果 =====================
train_results_df = pd.DataFrame({'Train_Actual': y_train, 'Train_Predicted': y_train_pred})
test_results_df  = pd.DataFrame({'Test_Actual': y_test, 'Test_Predicted': y_test_pred})

train_output_path = os.path.join(base_dir, 'XGB_train_results.xlsx')
test_output_path  = os.path.join(base_dir, 'XGB_test_results.xlsx')

train_results_df.to_excel(train_output_path, index=False)
test_results_df.to_excel(test_output_path, index=False)
print(f"Train results saved to {train_output_path}")
print(f"Test results saved to {test_output_path}")

# ===================== 10. 保存指标 =====================
metrics_df = pd.DataFrame({
    'Set':  ['Train', 'Test'],
    'R2':   [train_r2, test_r2],
    'RMSE': [train_rmse, test_rmse],
    'MSE':  [train_mse, test_mse],
    'MAE':  [train_mae, test_mae]
})
metrics_path = os.path.join(base_dir, 'XGB_metrics.xlsx')
metrics_df.to_excel(metrics_path, index=False)
print(f"Metrics saved to {metrics_path}")

# ===================== 11. 绘图函数定义 =====================
def plot_scatter(y_true, y_pred, r2, rmse, mse, mae, title, output_path):
    plt.figure(figsize=(7, 7), dpi=600)
    plt.scatter(y_true, y_pred, c='dodgerblue', label='Prediction',
                edgecolors='k', alpha=0.7)

    # 拟合线 y=a*x+b
    coef = np.polyfit(y_true, y_pred, 1)
    poly1d_fn = np.poly1d(coef)
    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    x_line = np.linspace(min_val, max_val, 200)
    plt.plot(x_line, poly1d_fn(x_line), color='red',alpha=0.5,  label='Fit Line')

    # 1:1 线
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray',alpha=0.45, label='1:1 Line')

    # 设置字体和大小
    font = {'family': 'Times New Roman', 'size': 18}
    plt.xlabel('Actual', fontdict=font)
    plt.ylabel('Predicted', fontdict=font)
    plt.title(title, fontdict={'family': 'Times New Roman', 'size': 18})

    # 标注指标
    text_str = (f'$R^2$ = {r2:.3f}\n'
                f'RMSE = {rmse:.3f}\n'
                f'MSE = {mse:.3f}\n'
                f'MAE = {mae:.3f}')
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
             fontsize=18, va='top',
             family='Times New Roman',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.legend(loc='lower right', fontsize=22, prop={'family': 'Times New Roman'})
    plt.xticks(fontsize=20, family='Times New Roman')
    plt.yticks(fontsize=20, family='Times New Roman')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

# ===================== 12. 绘制并保存散点图 =====================
plot_scatter(y_train, y_train_pred, train_r2, train_rmse, train_mse, train_mae,
             'XGBoost Training Set', os.path.join(base_dir, 'XGB_train_scatter.png'))

plot_scatter(y_test, y_test_pred, test_r2, test_rmse, test_mse, test_mae,
             'XGBoost Test Set', os.path.join(base_dir, 'XGB_test_scatter.png'))

print(f"✅ 散点图已保存至：{base_dir}")

# ===================== 13. 特征重要性 =====================
importances = xgb_best.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n================= 🌲 特征重要性（前10）=================")
for i in range(min(50, len(feature_names))):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# ===================== 14. 保存模型 =====================
model_path = os.path.join(base_dir, f'best_XGB_model_seed_{BEST_SEED}.joblib')
joblib.dump(xgb_best, model_path)
print(f"\n💾 模型已保存至: {model_path}")
