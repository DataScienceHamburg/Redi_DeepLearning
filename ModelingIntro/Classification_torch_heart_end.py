#%% packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns


#%% data import
heart = pd.read_csv('data/heart.csv')
heart.head()

#%% visualise the model
# sns.pairplot(heart, hue='HeartDisease')
sns.countplot(heart['HeartDisease'].astype(str))

#%% Categorical Feature Treatment
heart_dummies = pd.get_dummies(heart, dtype=float)
#%% Separate Independent and Dependent Variables
X = heart_dummies.drop('HeartDisease',axis=1)
y = heart_dummies['HeartDisease']




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
X_train_scaled.shape
# %%
X_test_scaled = scaler.transform(X_test)
#%% Dataset and Dataloader
BS = 64
class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.values.reshape(-1,1).astype(np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset = ClassificationDataset(X_train_scaled, y_train), batch_size=BS)
test_loader = DataLoader(dataset = ClassificationDataset(X_test_scaled, y_test), batch_size=BS)


#%% Model
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

#%% Instance of the model
input_dim = X.shape[1]
output_dim = 1
hidden_size = 25
model = ClassificationTorch(input_size=input_dim, hidden_size=hidden_size, output_size=output_dim)

# %% Mean Squared Error
loss_fun = nn.BCELoss()

#%% Optimizer
optimizer = torch.optim.Adam(model.parameters())

#%% perform training
losses = []
slope, bias = [], []
number_epochs = 500
#%% training loop
for epoch in range(number_epochs):
    loss_batch = []
    for X_batch, y_batch in train_loader:
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X_batch)

        # compute loss
        loss = loss_fun(y_pred, y_batch)
        # store loss
        loss_batch.append(loss.item())
        

        # backprop
        loss.backward()

        # update weights
        optimizer.step()

    # store epoch loss
    losses.append(np.mean(loss_batch))
    print(f'Epoch {epoch+1}/{number_epochs}, loss: {np.mean(loss_batch)}')

#%% Visualise the loss
sns.lineplot(x=list(range(len(losses))), y=np.array(losses))

#%% Create Predictions
model.eval()
with torch.inference_mode():
    y_pred_test_prob = model(torch.tensor(X_test_scaled).float()).numpy().flatten()
    y_pred_test = np.where(y_pred_test_prob > 0.5, 1, 0)

#%% Check the predictions
print(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")

#%% Visualise the predictions
sns.heatmap(pd.crosstab(pd.Series(y_test, name='True'), pd.Series(y_pred_test, name='Pred')), annot=True, fmt='d', cmap='Blues')

# %% model state dict
model.state_dict()
# %% save model state dict
torch.save(model.state_dict(), 'model_state_dict.pth')

# %% load a model
model = ClassificationTorch(input_size=input_dim, output_size=output_dim)
# model.state_dict()  # randomly initialized
model.load_state_dict(torch.load('model_state_dict.pth'))
model.state_dict()
# %%
