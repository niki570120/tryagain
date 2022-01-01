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


raw_data = pd.read_csv('./bridgeconbine.csv')
raw_data


# In[5]:


raw_data.info()


# In[6]:


from sklearn.model_selection import train_test_split
features = ['temperature', 'OW1', 'OW2', 'T1(1-3)','T1(2-4)','T2(1-3)','T2(2-4)','T3(1-3)','T3(2-4)','T4(1-3)','T4(2-4)','T5(1-3)','T5(2-4)','T6(1-3)','T6(2-4)','number'] 
X = raw_data[features]
y = raw_data['deflection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[7]:


from sklearn.linear_model import LinearRegression#線性回歸
clf1 = LinearRegression()

clf1.fit(X_train, y_train)


# In[8]:


clf1.predict(X)


# In[18]:


from sklearn.linear_model import LinearRegression
clf1 = LinearRegression()
clf1.fit(X, y)
print('y = {:.4f} * x + {:.4f}'.format(clf1.coef_[0], clf1.intercept_))


# In[19]:


# 求 20 個 x 點對應的 y 值
import pandas as pd

# Create a dataframe with an x column containing values from -10 to 10
df = pd.DataFrame ({'x': range(-0,36)})

# Add a y column by applying the solved equation to x
df['y'] = (0.3491*df['x'] + 579.2472)

#Display the dataframe
df


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

plt.plot(df.x, df.y, color="red", marker = "o", markersize=1)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()


# In[21]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(X, y)
x1=np.arange(y)
y1=0.3491 * x1+ 579.2472
plt.plot(x1, y1, 'r')


# In[22]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

X_train = np.c_[0.5, 1].T
y_train = [0.5, 1]
X_test = np.c_[0, 2].T

np.random.seed(0)

classifiers = dict(
    D=linear_model.LinearRegression(), T=linear_model.Ridge(alpha=0.2)
)

for name, clf1 in classifiers.items():
    fig, ax = plt.subplots(figsize=(4, 3))

    for _ in range(6):
        this_X = 0.1 * np.random.normal(size=(2, 1)) + X_train
        clf1.fit(this_X, y_train)

        ax.plot(X_test, clf1.predict(X_test), color="gray")
        ax.scatter(this_X, y_train, s=3, c="gray", marker="o", zorder=10)

    clf1.fit(X_train, y_train)
    ax.plot(X_test, clf1.predict(X_test), linewidth=2, color="blue")
    ax.scatter(X_train, y_train, s=30, c="red", marker="+", zorder=10)

    ax.set_title(name)
    ax.set_xlim(0, 2)
    ax.set_ylim((0, 1.6))
    ax.set_xlabel("X")
    ax.set_ylabel("y")

    fig.tight_layout()

plt.show()


# In[23]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(X_train,y_train)
x1=np.arange(0, 100)
y1=0.3491 * x1+579.2472
plt.plot(x1, y1, 'r')


# In[24]:


plt.show(close=None, block=None)


# In[25]:


# 預測#線性回歸
y_pred = clf1.predict(X_test)
y_pred


# In[26]:


plt.figure(figsize=(20, 10))
sns.boxplot('temperature', 'deflection', data=raw_data)
plt.title('deflection vs  temperature')


# In[27]:


plt.figure(figsize=(20, 10))
sns.boxplot('OW1', 'OW2', data=raw_data)
plt.title('OW1 vs OW2')


# In[28]:


plt.figure(figsize=(20, 10))
sns.boxplot('T1(1-3)', 'T1(2-4)', data=raw_data)
plt.title('T1(1-3) vs T1(2-4)')


# In[29]:


plt.figure(figsize=(20, 10))
sns.boxplot('T2(1-3)', 'T2(2-4)', data=raw_data)
plt.title('T2(1-3) vs T2(2-4)')


# In[30]:


plt.figure(figsize=(20, 10))
sns.boxplot('T3(1-3)', 'T3(2-4)', data=raw_data)
plt.title('T3(1-3) vs T3(2-4)')


# In[31]:


plt.figure(figsize=(20, 10))
sns.boxplot('T4(1-3)', 'T4(2-4)', data=raw_data)
plt.title('T4(1-3) vs T4(2-4)')


# In[32]:


plt.figure(figsize=(20, 10))
sns.boxplot('T5(1-3)', 'T5(2-4)', data=raw_data)
plt.title('T5(1-3) vs T5(2-4)')


# In[33]:


plt.figure(figsize=(20, 10))
sns.boxplot('T6(1-3)', 'T6(2-4)', data=raw_data)
plt.title('T6(1-3) vs T6(2-4)')


# In[34]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='date', data=raw_data)
plt.title('bridgedata')


# In[35]:


plt.figure(figsize=(12, 10))
ax = sns.scatterplot(x="date", y="deflection", data=raw_data)
plt.title('date vs deflection of bridge')


# In[36]:


plt.figure(figsize=(20, 10))
sns.pointplot('temperature', 'deflection', data=raw_data, palette='Set2')
plt.title('Variation of Height for Male Athletes over time')


# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv('./bridgeconbine.csv')
df


# In[5]:


from sklearn.model_selection import train_test_split

X = df['temperature']
y = df['deflection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[20]:


import numpy as np
import array

X=np.array.reshape(1,-1)


# In[16]:


from sklearn.linear_model import LinearRegression#線性回歸
clf = LinearRegression()
clf.fit(X_train, y_train)


# In[ ]:





# In[ ]:





# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

plt.plot(X, y, color="grey", marker = "o", markersize=6)
plt.xlabel('X')
plt.ylabel('y')
plt.grid()
plt.show()


# In[ ]:




