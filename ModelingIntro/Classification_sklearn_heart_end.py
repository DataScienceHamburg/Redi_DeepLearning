#%% packages
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#%% data import
# We will predict heart disease events based on 11 independent features.

# - Age: age of the patient [years]
# - Sex: sex of the patient [M: Male, F: Female]
# - ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# - RestingBP: resting blood pressure [mm Hg]
# - Cholesterol: serum cholesterol [mm/dl]
# - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# - RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
# - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
# - ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
# - Oldpeak: oldpeak = ST [Numeric value measured in depression]
# - ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]

# Dependent Feature:
# - HeartDisease: output class [1: heart disease, 0: Normal]

heart = pd.read_csv('data/heart.csv')
heart.head()

#%% Correlation Matrix
sns.heatmap(heart.select_dtypes(include='number').corr(), annot=True)
#%% visualise the model
sns.pairplot(heart, hue='HeartDisease')

#%% column info
heart.info()
heart.describe()
#%% Categorical Feature Treatment
heart_dummies = pd.get_dummies(heart, dtype=float)

#%% Separate Independent and Dependent Variables
X = heart_dummies.drop('HeartDisease',axis=1)
y = heart_dummies['HeartDisease']

#%% Naive Baseline
from collections import Counter
target_cnt = Counter(heart['HeartDisease'])
target_cnt

naive_accuracy = target_cnt[max(target_cnt)] / len(heart['HeartDisease']) * 100
print(f"Pure Guessing results in an accuracy of {naive_accuracy}.")

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
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# %% Predictions
y_pred_test = model.predict(X_test_scaled)
# %% Check the predictions
print(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")

