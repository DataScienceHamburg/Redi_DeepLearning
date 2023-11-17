#%% packages
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import seaborn as sns
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

#%% Dataset and DataLoader
BS = 64
class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.values.reshape(-1,1).astype(np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
#%% Instances for train and test sets and loaders
train_set = ClassificationDataset(X_train_scaled, y_train)
train_loader = DataLoader(dataset = train_set, batch_size=BS)
test_set = ClassificationDataset(X_test_scaled, y_test)
test_loader = DataLoader(dataset = test_set, batch_size=BS)

# %% Model
class ClassificationTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassificationTorch, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear_in(x)
        x = torch.relu(x)
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        return x


# %%
input_size = X_train_scaled.shape[1]
hidden_size = 32
output_size = 1
model = ClassificationTorch(input_size, hidden_size, output_size)
# %% Loss Function and Optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# %%
losses = []
number_epochs = 400

# %%
for epoch in range(number_epochs):
    loss_batch = []
    for X_batch, y_batch in train_loader:
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X_batch)

        # compute loss
        loss = loss_fn(y_pred, y_batch)
        # store loss
        loss_batch.append(loss.item())
        # backprop
        loss.backward()

        # update weights
        optimizer.step()

    # store epoch loss
    losses.append(np.mean(loss_batch))
    print(f'Epoch {epoch+1}/{number_epochs}, loss: {np.mean(loss_batch)}')

# %% Visualize the loss
sns.lineplot(x=range(number_epochs), y=losses)

# %% Model Evaluation
model.eval()
with torch.inference_mode():
    y_pred_test_prob = model(torch.tensor(X_test_scaled).float()).numpy().flatten()
    threshold = 0.5
    y_pred_test = np.where(y_pred_test_prob > threshold, 1, 0)

#%% Check the predictions
print(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")

# %%
