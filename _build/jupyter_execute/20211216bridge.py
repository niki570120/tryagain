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


df = pd.read_csv('./bridgeconbine121603.csv')
df


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt 
plt.figure(figsize=(12, 10))
ax = sns.scatterplot(x="temperature", y="deflection", data=df)
plt.title('temperature vs deflection of Baishihu Suspension Bridge')


# In[6]:


sns.distplot(df['deflection'],kde=0)


# In[7]:


from sklearn.model_selection import train_test_split
features = ['temperature', 'OW1', 'OW2', 'T1(1-3)','T1(2-4)','T2(1-3)','T2(2-4)','T3(1-3)','T3(2-4)','T4(1-3)','T4(2-4)','T5(1-3)','T5(2-4)','T6(1-3)','T6(2-4)','number','SO1(1.5)','SO1(2.5)','SO1(3.5)','SO2(0.5)','SO2(10.5)','SO2(11.0)'] 

X = df[features]
y = df['deflection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[8]:


from sklearn.linear_model import LinearRegression#線性回歸
clf1 = LinearRegression()

clf1.fit(X_train, y_train)


# In[9]:


clf1.intercept_


# In[10]:


clf1.coef_


# In[11]:


y_pred = clf1.predict(X_test)
y_pred


# In[12]:


# 幫模型打分數
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[13]:


# 幫模型打分數
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[14]:


from sklearn.linear_model import LinearRegression
clf1 = LinearRegression()
clf1.fit(X, y)
print('y = {:.4f} * x + {:.4f}'.format(clf1.coef_[0], clf1.intercept_))


# In[15]:


clf1.fit(X, y)


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
pcr.fit(X_train, y_train)
pca = pcr.named_steps["pca"]  # retrieve the PCA step of the pipeline

pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].scatter(pca.transform(X_test), y_test, alpha=0.3, label="ground truth")
axes[0].scatter(
    pca.transform(X_test), pcr.predict(X_test), alpha=0.3, label="predictions"
)
axes[0].set(
    xlabel="Projected data onto first PCA component", ylabel="y", title="PCR / PCA"
)
axes[0].legend()
axes[1].scatter(pls.transform(X_test), y_test, alpha=0.3, label="ground truth")
axes[1].scatter(
    pls.transform(X_test), pls.predict(X_test), alpha=0.3, label="predictions"
)
axes[1].set(xlabel="Projected data onto first PLS component", ylabel="y", title="PLS")
axes[1].legend()
plt.tight_layout()
plt.show()


# In[17]:


print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")
print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")


# In[18]:


pca_2 = make_pipeline(PCA(n_components=1), LinearRegression())
pca_2.fit(X_train, y_train)
print(f"PCR r-squared with 2 components {pca_2.score(X_test, y_test):.3f}")


# In[19]:


import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# In[20]:


from sklearn.linear_model import LinearRegression#線性回歸
clf2 = LinearRegression()

clf2.fit(X_train, y_train)


# In[21]:


from sklearn.linear_model import LinearRegression
clf2 = LinearRegression()
clf2.fit(X, y)
print('y = {:.4f} * x + {:.4f}'.format(clf2.coef_[0], clf2.intercept_))


# In[22]:


from sklearn.model_selection import train_test_split
features02 = ['temperature','T1(1-3)','T1(2-4)','T2(1-3)','T2(2-4)','T3(1-3)','T3(2-4)','T4(1-3)','T4(2-4)','T5(1-3)','T5(2-4)','T6(1-3)','T6(2-4)'] 

X = df[features02]
y = df['deflection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[23]:


from sklearn.linear_model import LinearRegression#線性回歸
clf3 = LinearRegression()

clf3.fit(X_train, y_train)


# In[24]:


from sklearn.linear_model import LinearRegression
clf3 = LinearRegression()
clf3.fit(X, y)
print('y = {:.4f} * x + {:.4f}'.format(clf3.coef_[0], clf3.intercept_))


# In[25]:


from sklearn.model_selection import train_test_split
features04 = ['OW1', 'OW2','SO1(1.5)','SO1(2.5)','SO1(3.5)','SO2(0.5)','SO2(10.5)','SO2(11.0)'] 

X = df[features04]
y = df['deflection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[26]:


from sklearn.linear_model import LinearRegression#線性回歸
clf4 = LinearRegression()

clf4.fit(X_train, y_train)


# In[27]:


from sklearn.linear_model import LinearRegression
clf4 = LinearRegression()
clf4.fit(X, y)
print('y = {:.4f} * x + {:.4f}'.format(clf4.coef_[0], clf4.intercept_))


# In[28]:


from sklearn.model_selection import train_test_split
features05 = ['T1(1-3)','T1(2-4)','T2(1-3)','T2(2-4)','T3(1-3)','T3(2-4)','T4(1-3)','T4(2-4)','T5(1-3)','T5(2-4)','T6(1-3)','T6(2-4)'] 

X = df[features05]
y = df['deflection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[29]:


from sklearn.linear_model import LinearRegression#線性回歸
clf5 = LinearRegression()

clf5.fit(X_train, y_train)


# In[30]:


from sklearn.linear_model import LinearRegression
clf5 = LinearRegression()
clf5.fit(X, y)
print('y = {:.4f} * x + {:.4f}'.format(clf5.coef_[0], clf5.intercept_))


# In[31]:


df_num = df.drop(columns=["date"])

fig, axes = plt.subplots(len(df_num.columns)//3, 3, figsize=(15, 8))
i = 0
for triaxis in axes:
    for axis in triaxis:
        df_num.hist(column = df_num.columns[i], ax=axis)
        i = i+1


# In[ ]:




