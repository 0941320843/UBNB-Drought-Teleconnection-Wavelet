

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

# Load Dataset

df = pd.read_csv(r"C:\Users\user\Downloads\Metrology data all of my thesis (1).csv")


df['Date'] = pd.to_datetime(dict(year=df['YEAR'], month=df['MO'], day=df['DY']))

# Features and Target

X = df[['RH', 'T2MDEW', 'TMIN', 'TMAX', 'WSPE', 'cape', 'sunshine', 'sst']]
y = df['Rain fall']


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# Model + Grid Search
params = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'num_leaves': [15, 31, 63]
}

grid = GridSearchCV(
    estimator=LGBMRegressor(random_state=42, verbosity=-1),
    param_grid=params,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Fit model
grid.fit(X_train, y_train)


best_lgb = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

# Predictions


y_pred = best_lgb.predict(X_test)
# Evaluation

print("\nModel Performance:")
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Plot: Observed vs Predicted

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)

plt.xlabel("Observed Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("LightGBM: Observed vs Predicted Rainfall")

# ideal line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.tight_layout()
plt.show()