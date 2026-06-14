# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:19:22 2026

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

# ==========================
# Load Data
# ==========================

df = pd.read_csv("C:\\Users\\user\\Downloads\\Metrology data all of my thesis (1).csv")
# Figure 3
# Correlation Heatmap
# ==========================

variables = [
    'Rain fall',
    'RH',
    'T2MDEW',
    'TMIN',
    'TMAX',
    'WSPE',
    'cape',
    'sunshine',
    'sst'
]

corr_matrix = df[variables].corr(method='pearson')

plt.figure(figsize=(10,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    fmt='.2f'
)

plt.title(
    'Correlation Heatmap of Features with Rainfall',
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

# ==========================
# Figure 4
# Random Forest Feature Importance
# ==========================

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

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance Scores")
print(importance)

plt.figure(figsize=(8,6))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title(
    'Relative Importance of Feature Variables for Rainfall Estimation',
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()
plt.show()