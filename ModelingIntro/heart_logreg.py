#%% packages
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# %% Load the data
heart = pd.read_csv('data/heart.csv')
heart.head()

# %%
sns.heatmap(heart.select_dtypes(include='number').corr(), annot=True)
# %%
sns.pairplot(heart, hue='HeartDisease')
# %% Categorical Feature Treatment
heart_dummies = pd.get_dummies(heart, dtype=float)
heart_dummies.head()

# %% Separate Independent (X) and Dependent Variables (y)
X = heart_dummies.drop('HeartDisease',axis=1)
y = heart_dummies['HeartDisease']

# %% Separate Train and Test Sets
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Baseline Classifier

import numpy as np
cnt_train = Counter(y_train)
baseline_acc = cnt_train[max(cnt_train)] / len(y_train) * 100
print(f"Pure Guessing results in an accuracy of {np.round(baseline_acc, 1)}%")

#%% Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %% Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# %% Predictions
y_pred_test = model.predict(X_test_scaled)
# %% Confusion Matrix
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, y_pred_test),annot=True)

# %% Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_test)
print(f"Accuracy: {np.round(accuracy_score(y_test, y_pred_test), 1)*100}%")
# %%
