#%% packages
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
#%% data import
cement_file = 'data/Concrete_Data_Yeh.csv'
cement = pd.read_csv(cement_file)
cement.head()

#%% visualise the model
sns.pairplot(cement)

#%% Separate Independent and Dependent Variables
X = cement.drop('csMPa',axis=1)
y = cement['csMPa']

#%% Train Test Split
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# %% Scale the data
scaler = StandardScaler()
scaler.fit(X_train)

# %% Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled
# %%
X_test_scaled = scaler.transform(X_test)
# %% Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# %% Predictions
y_pred_test = model.predict(X_test_scaled)
# %% Check the predictions
print(f"R2: {r2_score(y_test, y_pred_test)}")

# %%
