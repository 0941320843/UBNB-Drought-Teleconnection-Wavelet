
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# Load Dataset

df = pd.read_csv(r"C:\Users\user\Downloads\Metrology data all of my thesis (1).csv"
)


df['Date'] = pd.to_datetime(
    dict(
        year=df['YEAR'],
        month=df['MO'],
        day=df['DY']
    )
)


X = df[
    [
        'RH',
        'T2MDEW',
        'TMIN',
        'TMAX',
        'WSPE',
        'cape',
        'sunshine',
        'sst'
    ]
]

y = df['Rain fall']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Grid Search CV

params = {
    'max_depth': [3,5,7,10,15,20,None],
    'min_samples_split': [2,5,10,20],
    'min_samples_leaf': [1,2,4,8]
}

grid = GridSearchCV(
    estimator=DecisionTreeRegressor(
        random_state=42
    ),
    param_grid=params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid.fit(X_train, y_train)


# Best Model


best_tree = grid.best_estimator_

print("\nBest Parameters")
print(grid.best_params_)

# Predictions


y_pred = best_tree.predict(X_test)

# Statistics


r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

train_r2 = best_tree.score(
    X_train,
    y_train
)

test_r2 = best_tree.score(
    X_test,
    y_test
)

print("\nModel Performance")
print(f"Train R² = {train_r2:.4f}")
print(f"Test  R² = {test_r2:.4f}")
print(f"RMSE      = {rmse:.4f}")
print(f"MAE       = {mae:.4f}")


plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.65,
    color='black'
)


min_val = min(
    y_test.min(),
    y_pred.min()
)

max_val = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_val,max_val],
    [min_val,max_val],
    'r--',
    linewidth=2,
    label='1:1 Line'
)

plt.xlabel(
    'Observed Rainfall (mm)',
    fontsize=12
)

plt.ylabel(
    'Predicted Rainfall (mm)',
    fontsize=12
)

plt.title(
    f'RT Model (R² = {r2:.3f})',
    fontsize=13,
    fontweight='bold'
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    'Figure5_Optimized_RT.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()