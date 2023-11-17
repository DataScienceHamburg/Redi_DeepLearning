#%% packages
import pandas as pd
import seaborn as sns
# %% Load the data
heart = pd.read_csv('data/heart.csv')
heart.head()

# %%
sns.heatmap(heart.select_dtypes(include='number').corr(), annot=True)
# %%
