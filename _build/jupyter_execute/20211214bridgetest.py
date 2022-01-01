#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('./bridgeconbine.csv')
df


# In[3]:


df.info()


# In[5]:


df.describe()


# In[7]:


from sklearn.model_selection import train_test_split
features = ['temperature', 'OW1', 'OW2', 'T1(1-3)','T1(2-4)','T2(1-3)','T2(2-4)','T3(1-3)','T3(2-4)','T4(1-3)','T4(2-4)','T5(1-3)','T5(2-4)','T6(1-3)','T6(2-4)','number'] 
X = df[features]
y = df['deflection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[8]:


from sklearn.linear_model import LinearRegression#線性回歸
clf1 = LinearRegression()

clf1.fit(X_train, y_train)


# In[9]:


from sklearn.linear_model import LinearRegression
clf1 = LinearRegression()
clf1.fit(X, y)
print('y = {:.4f} * x + {:.4f}'.format(clf1.coef_[0], clf1.intercept_))


# In[10]:


from sklearn.neighbors import NearestNeighbors
clf2 = NearestNeighbors()

clf2.fit(X_train, y_train)


# In[16]:


X_train


# In[17]:


X_test


# In[18]:


df_train, df_test = train_test_split(df, test_size=0.25, random_state=0)
print('Size of train dataframe: ', df_train.shape[0])
print('Size of test dataframe: ', df_test.shape[0])


# In[21]:


# Create a flag for label masking
df_train['temperature'] = True
df_train.loc[df_train.sample(frac=0.05, random_state=0).index, 'temperature'] = False

# Create a new target colum with labels. The 1's and 0's are original labels and -1 represents unlabeled (masked) data
df_train['Dependents_Target']=df_train.apply(lambda x: x['OW1'] if x['temperature']==False else -1, axis=1)

# Show target value distribution
print('Target Value Distribution:')
print(df_train['Dependents_Target'].value_counts())


# In[25]:


# Create a scatter plot
fig = df.deflection(df_train, x='temperature', y='MntWines', opacity=1, color=df_train['Dependents_Target'].astype(str),
                 color_discrete_sequence=['lightgrey', 'red', 'blue'],
                )

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='white', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='white', 
                 showline=True, linewidth=1, linecolor='white')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='white', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='white', 
                 showline=True, linewidth=1, linecolor='white')

# Set figure title
fig.update_layout(title_text="Marketing Campaign Training Data - Labeled vs. Unlabeled")

# Update marker size
fig.update_traces(marker=dict(size=5))

fig.show()


# In[ ]:





# In[ ]:




