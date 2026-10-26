# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 参数设置 =====================
random_states = range(1, 101) # 从1到100逐一迭代
results = []
output_path = r"C:/Users/变量建模结果.xlsx"

# ===================== 2. 加载与预处理数据 =====================
file_path = r'C:/Users/建模S2.xlsx'
data = pd.read_excel(file_path)

if data.isnull().values.any():
    print("⚠️ 检测到缺失值，正在用中位数填充...")
    data = data.fillna(data.median(numeric_only=True))

y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values

# ===================== 3. 定义参数网格 =====================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3,5,7,10],
    'learning_rate': [0.01,0.05, 0.1],
    'subsample': [0.6,0.8,1.0],
    'colsample_bytree': [0.6,0.8,1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3]
}

# ===================== 4. 主循环：逐一迭代随机数 =====================
for seed in random_states:
    print(f"\n================= 🎯 当前随机数种子: {seed} =================")
    np.random.seed(seed)

    # 4.1 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    # 4.2 定义模型与交叉验证
    xgb = XGBRegressor(
        random_state=seed,
        objective='reg:squarederror',
        n_jobs=-1
    )
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='r2',
        cv=kfold,
        n_jobs=-1,
        verbose=0
    )

    # 4.3 模型训练
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 4.4 训练集评估
    y_train_pred = best_model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # 4.5 测试集评估
    y_test_pred = best_model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # 4.6 保存当前结果
    results.append({
        "Random_State": seed,
        "Best_Params": str(grid_search.best_params_),
        "CV_R2": round(grid_search.best_score_, 4),
        "Train_R2": round(r2_train, 4),
        "Train_RMSE": round(rmse_train, 4),
        "Test_R2": round(r2_test, 4),
        "Test_RMSE": round(rmse_test, 4)
    })

    print(f"✅ 完成 seed={seed}: Test R²={r2_test:.4f}, RMSE={rmse_test:.4f}")

# ===================== 5. 导出结果 =====================
df_results = pd.DataFrame(results)

# 添加平均值与标准差行
mean_row = {
    "Random_State": "Mean",
    "Best_Params": "-",
    "CV_R2": df_results["CV_R2"].mean(),
    "Train_R2": df_results["Train_R2"].mean(),
    "Train_RMSE": df_results["Train_RMSE"].mean(),
    "Test_R2": df_results["Test_R2"].mean(),
    "Test_RMSE": df_results["Test_RMSE"].mean()
}
std_row = {
    "Random_State": "Std",
    "Best_Params": "-",
    "CV_R2": df_results["CV_R2"].std(),
    "Train_R2": df_results["Train_R2"].std(),
    "Train_RMSE": df_results["Train_RMSE"].std(),
    "Test_R2": df_results["Test_R2"].std(),
    "Test_RMSE": df_results["Test_RMSE"].std()
}
df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)

df_results.to_excel(output_path, index=False)
print(f"\n💾 所有 1~100 随机数结果已导出至：{output_path}")

# ===================== 6. 汇总统计 =====================
print("\n================= 📈 综合统计 =================")
print(f"平均测试 R²: {df_results.loc[df_results['Random_State']=='Mean', 'Test_R2'].values[0]:.4f}")
print(f"标准差 R²: {df_results.loc[df_results['Random_State']=='Std', 'Test_R2'].values[0]:.4f}")
print(f"平均测试 RMSE: {df_results.loc[df_results['Random_State']=='Mean', 'Test_RMSE'].values[0]:.4f}")
print(f"标准差 RMSE: {df_results.loc[df_results['Random_State']=='Std', 'Test_RMSE'].values[0]:.4f}")

# ===================== 7. 保存最优模型 =====================
best_row = df_results.iloc[:-2].loc[df_results["Test_R2"].idxmax()]
best_seed = int(best_row["Random_State"])
print(f"\n🌟 最优随机种子: {best_seed}, 对应测试R²={best_row['Test_R2']:.4f}")

# 重新训练最优模型并保存
np.random.seed(best_seed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=best_seed
)
xgb = XGBRegressor(random_state=best_seed, objective='reg:squarederror', n_jobs=-1)
kfold = KFold(n_splits=5, shuffle=True, random_state=best_seed)
grid_search = GridSearchCV(
    estimator=xgb, param_grid=param_grid, scoring='r2', cv=kfold, n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

model_path = f"best_XGB_model_seed_{best_seed}.joblib"
joblib.dump(best_model, model_path)
print(f"💾 最优模型已保存至: {model_path}")
