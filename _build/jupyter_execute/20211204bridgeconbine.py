#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df=pd.read_csv('./bridgeconbine.csv')
df


# In[57]:


df.describe()


# In[58]:


df.info()


# In[59]:


df.dtypes


# In[ ]:





# In[60]:


df1=pd.read_excel('./bridgeconbine.xlsx')
df1.head()


# In[ ]:





# In[61]:


plt.figure(figsize=(20, 10))
sns.boxplot('temperature', 'deflection', data=df1)
plt.title('deflection vs  temperature')


# In[62]:


plt.figure(figsize=(20, 10))
sns.boxplot('temperature', 'number', data=df1)
plt.title('number vs  temperature')


# In[63]:


from sklearn.model_selection import train_test_split
features = ['temperature', 'OW1', 'OW2', 'T1(1-3)','T1(2-4)','T2(1-3)','T2(2-4)','T3(1-3)','T3(2-4)','T4(1-3)','T4(2-4)','T5(1-3)','T5(2-4)','T6(1-3)','T6(2-4)','number'] 
X = df1[features]
y = df1['deflection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[64]:


from sklearn.linear_model import LinearRegression#線性回歸
regr=LinearRegression()
regr.fit(X_train, y_train)


# In[34]:


from sklearn.linear_model import LinearRegression#線性回歸
clf1 = LinearRegression()

clf1.fit(X_train, y_train)


# In[ ]:





# In[35]:


clf1.predict(X)


# In[36]:


plt.scatter(clf1.predict(X),y)


# In[37]:


X_train=np.arange(1, len(y_train)+1)


# In[38]:


plt.plot(X_train, y_train, marker='o', markersize=6)


# In[39]:


from sklearn.linear_model import LinearRegression#線性回歸
clf1 = LinearRegression() 
clf1.fit(X_train, y_train)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,clf1.predict(X_train), color = 'green')
plt.title('salary vs yearExp (Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


# In[40]:


plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, clf1.predict(X_test),color='blue',linewidth=1)
X_train[:,0]
plt.show()


# In[41]:


regplot(x,clf1.predict(X)) 


# In[42]:


import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# 隨機產生一個特徵的X與輸出y
X, y = make_data(100)

# 建立 SGDRegressor 並設置超參數
clf1 = SGDRegressor(max_iter=100)
# 訓練模型
clf1.fit(X, y)
# 建立測試資料
x_test = np.linspace(-0.05,1,500)[:,None]
# 預測測試集
y_test=clf1.predict(x_test)
# 預測訓練集
y_pred=clf1.predict(X)
# 視覺化預測結果
plt.scatter(X,y)
plt.plot(x_test.ravel(),y_test, color="#d62728")
plt.xlabel('x')
plt.ylabel('y')
plt.text(0, 10, 'Loss(MSE)=%.3f' % mean_squared_error(y_pred, y), fontdict={'size': 15, 'color':  'red'})
plt.show()


# In[43]:


# 超參數(Hyperparameters)
x_start = 5     # 起始權重
epochs = 15     # 執行週期數 
lr = 0.3        # 學習率 

# 梯度下降法 
# *** Function 可以直接當參數傳遞 ***
w = clf1.predict(X)(x_start, c1f1, epochs, lr=lr) 

color = 'red'    

plt.figure(figsize=(12,8))
t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')

# 設定中文字型
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msjhbd.ttc", size=20)   
# 矯正負號
plt.rcParams['axes.unicode_minus'] = False

plt.title('梯度下降法', fontproperties=font)
plt.xlabel('w', fontsize=20)
plt.ylabel('Loss', fontsize=20)

color = list('rgbymr')  # 切線顏色   
line_offset=2           #切線長度
for i in range(5, -1, -1):
    # 取相近兩個點，畫切線(tangent line)
    z=np.array([i+0.001, i])
    vec=np.vectorize(func)
    cls = np.polyfit(z, vec(z), deg=1)
    p = np.poly1d(cls)
    
    # 畫切線
    x=np.array([i+line_offset, i-line_offset])
    y=np.array([(i+line_offset)*p[1]+p[0], (i-line_offset)*p[1]+p[0]])
    plt.plot(x, y, c=color[i-1])    

plt.show()


# In[ ]:





# In[44]:


regions = pd.read_csv('./bridgeconbine.csv')


# In[45]:


regions.head(10)


# In[46]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(df['OW1'])
plt.title('Bridge water ele.')


# In[47]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(df['OW2'])
plt.title('Bridge water ele.')


# In[48]:


df.rename(columns = {'日期': 'date', '撓度': 'deflection','溫度': 'temperature'}, inplace=True)
df.head()


# In[49]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='date', data=df)
plt.title('bridgeconbine')


# In[50]:


plt.figure(figsize=(12, 10))
ax = sns.scatterplot(x="date", y="deflection", data=df)
plt.title('date vs deflection of bridge')


# In[51]:


sns.histplot(data=df, x='deflection')


# In[52]:


sns.histplot(data=df, x='T1(1-3)')


# In[53]:


df.isnull().sum()


# In[ ]:





# In[ ]:





# In[54]:


from sklearn.model_selection import train_test_split
features = ['temperature', 'OW1', 'OW2', 'T1(1-3)','T1(2-4)','T2(1-3)','T2(2-4)','T3(1-3)','T3(2-4)','T4(1-3)','T4(2-4)','T5(1-3)','T5(2-4)','T6(1-3)','T6(2-4)']

X = df[features]
y = df['deflection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[55]:


from sklearn.linear_model import LinearRegression#線性回歸
clf1 = LinearRegression()

clf1.fit(X_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[64]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(goldMedals['City'])
plt.title('Distribution of Gold Medals')


# In[65]:


goldMedals['ID'][goldMedals['Age'] > 45].count()


# In[66]:


masterDisciplines = goldMedals['Sport'][goldMedals['Age'] > 45]


# In[67]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(masterDisciplines)
plt.title('Gold Medals for Athletes Over 45')


# In[68]:


goldMedals['ID'][goldMedals['Age'] <20].count()


# In[69]:


masterDisciplines = goldMedals['Sport'][goldMedals['Age'] <20]


# In[70]:


masterDisciplines


# In[71]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(masterDisciplines)
plt.title('Gold Medals for Athletes MAX 20')


# In[72]:


menInOlympics = merged[(merged.Sex == 'M') & (merged.Season == 'Summer')]


# In[73]:


menInOlympics.head(10)


# In[74]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=menInOlympics)
plt.title('men medals per edition of the Games')


# In[75]:


menInOlympics.loc[menInOlympics['Year'] == 1900].head(10)


# In[76]:


menInOlympics['ID'].loc[menInOlympics['Year'] == 1900].count()


# In[77]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=menInOlympics)
plt.title('men medals per edition of the Games')


# In[30]:


womenInOlympics = merged[(merged.Sex == 'F') & (merged.Season == 'Summer')]
womenInOlympics.head(10)


# In[31]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=womenInOlympics)
plt.title('Women medals per edition of the Games')


# In[32]:


womenInOlympics.loc[womenInOlympics['Year'] == 1900].head(10)


# In[33]:


womenInOlympics['ID'].loc[womenInOlympics['Year'] == 1900].count()


# In[34]:


goldMedals.region.value_counts().reset_index(name='Medal').head(5)


# In[35]:


totalGoldMedals = goldMedals.region.value_counts().reset_index(name='Medal').head(5)
g = sns.catplot(x="index", y="Medal", data=totalGoldMedals,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_xlabels("Top 5 countries")
g.set_ylabels("Number of Medals")
plt.title('Medals per Country')


# In[42]:


goldMedalsUSA = goldMedals.loc[goldMedals['NOC'] == 'USA']


# In[43]:


goldMedalsUSA.Event.value_counts().reset_index(name='Medal').head(20)


# In[44]:


goldMedalsTPE = goldMedals.loc[goldMedals['NOC'] == 'TPE']


# In[45]:


goldMedalsTPE.Event.value_counts().reset_index(name='Medal').head(20)


# In[54]:


goldMedalsJAPAN = goldMedals.loc[goldMedals['NOC'] == 'JAPAN']


# In[55]:


goldMedalsJAPAN.Event.value_counts().reset_index(name='Medal').head()


# In[56]:


goldMedalsTPE = goldMedals.loc[goldMedals['NOC'] == 'TPE']


# In[57]:


goldMedalsTPE.Event.value_counts().reset_index(name='Medal').head()


# In[58]:


goldMedalsFIN = goldMedals.loc[goldMedals['NOC'] == 'FIN']


# In[59]:


goldMedalsFIN.Event.value_counts().reset_index(name='Medal').head(20)


# In[60]:


FootballmenGoldFIN = goldMedalsFIN.loc[(goldMedalsFIN['Sport'] == 'Football') & (goldMedalsFIN['Sex'] == 'M')].sort_values(['Year'])


# In[61]:


FootballmenGoldFIN.head(15)


# In[62]:


goldMedalsUSA = goldMedals.loc[goldMedals['NOC'] == 'USA']


# In[63]:


goldMedalsUSA.Event.value_counts().reset_index(name='Medal').head(20)


# In[64]:


basketballGoldUSA = goldMedalsUSA.loc[(goldMedalsUSA['Sport'] == 'Basketball') & (goldMedalsUSA['Sex'] == 'M')].sort_values(['Year'])


# In[65]:


basketballGoldUSA.head(15)


# In[66]:


groupedBasketUSA = basketballGoldUSA.groupby(['Year']).first()
groupedBasketUSA


# In[67]:


groupedBasketUSA['ID'].count()


# In[68]:


goldMedals.head()


# In[69]:


goldMedals.info()


# In[70]:


notNullMedals = goldMedals[(goldMedals['Height'].notnull()) & (goldMedals['Weight'].notnull())]


# In[71]:


notNullMedals.head()


# In[72]:


notNullMedals.info()


# In[80]:


plt.figure(figsize=(12, 10))
ax = sns.scatterplot(x="Height", y="Age", data=notNullMedals)
plt.title('Height vs Weight of Olympic Medalists')


# In[81]:


notNullMedals.loc[notNullMedals['Weight'] > 150]


# In[82]:


MenOverTime = merged[(merged.Sex == 'M') & (merged.Season == 'Summer')]
WomenOverTime = merged[(merged.Sex == 'F') & (merged.Season == 'Summer')]


# In[83]:


MenOverTime.head()


# In[84]:


part = MenOverTime.groupby('Year')['Sex'].value_counts()
plt.figure(figsize=(20, 10))
part.loc[:,'M'].plot()
plt.title('Variation of Male Athletes over time')


# In[85]:


part = WomenOverTime.groupby('Year')['Sex'].value_counts()
plt.figure(figsize=(20, 10))
part.loc[:,'F'].plot()
plt.title('Variation of Female Athletes over time')


# In[86]:


plt.figure(figsize=(20, 10))
sns.boxplot('Year', 'Age', data=MenOverTime)
plt.title('Variation of Age for Male Athletes over time')


# In[87]:


plt.figure(figsize=(20, 10))
sns.boxplot('Year', 'Age', data=WomenOverTime)
plt.title('Variation of Age for Female Athletes over time')


# In[88]:


plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Weight', data=MenOverTime)
plt.title('Variation of Weight for Male Athletes over time')


# In[89]:


plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Weight', data=WomenOverTime)
plt.title('Variation of Weight for Female Athletes over time')


# In[90]:


df1 = df.query('NOC == "TPE"')#台灣隊資料
df1.head(10)


# In[91]:


df1.shape


# In[92]:


df1.info()


# In[93]:


df2 = df1.dropna()
df2.shape


# In[94]:


df2.Medal.value_counts()


# In[95]:


df2.info()


# In[96]:


df2.pivot_table(values='Medal', index='Year', columns='Sport', aggfunc='count', fill_value=0)


# In[97]:


df2.pivot_table(values='ID', index='Year', columns='Medal', aggfunc='count', fill_value=0)


# In[100]:


plt.figure(figsize=(12, 10))
ax = sns.scatterplot(x="Height", y="Weight", data=df2)
plt.title('Height vs Weight of Olympic Medalists TPE')


# In[ ]:





# In[102]:


plt.figure(figsize=(12, 10))
ax = sns.scatterplot(x="Age", y="Weight", data=df2)
plt.title('Height vs Weight of Olympic Medalists TPE')


# In[103]:


import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

#import plotly.offline as py
#py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

import warnings
warnings.filterwarnings('ignore')


# In[104]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='grey',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df1['Sport'])


# In[105]:


y0 = df2[df2['Sex']=='M']['Age']
y1 = df2[df2['Sex']=='F']['Age']

trace0 = go.Box(
    y=y0,
    name="Age Distribution for Male")
trace1 = go.Box(
    y=y1,
    name="Age Distribution for Female")
data = [trace0, trace1]
iplot(data)


# In[106]:


df_medals=df2[df2['Medal']=='Gold']

cnt_srs = df_medals['Team'].value_counts().head(20)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="blue",
        #colorscale = 'Blues',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Top 20 countries with Maximum Gold Medals'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="medal")  


# In[107]:


cnt_srs = df2['Sport'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Most Popular Sport'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="sport")


# In[108]:


df_medals=df[df['Medal']=='Gold']

cnt_srs = df_medals['Team'].value_counts().head(20)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="blue",
        #colorscale = 'Blues',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Top 20 countries with Maximum Gold Medals'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="medal")  


# In[110]:


cnt_srs = df['Sport'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Most Popular Sport'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="sport")


# In[111]:


df_usa=df[(df['Team']=='United States')]
df_usa_medal=df_usa[df_usa['Medal']=='Gold']

medal_map = {'Gold':1}
df_usa_medal['Medal'] = df_usa_medal['Medal'].map(medal_map)

df_usa_sport=df_usa_medal.groupby(['Sport'],as_index=False)['Medal'].agg('sum')

df_usa_sport=df_usa_sport.sort_values(['Medal'],ascending=False)

df_usa_sport=df_usa_sport.head(10)

colors = ['#91BBF4', '#91F4F4', '#F79981', '#F7E781', '#C0F781','rgb(32,155,160)', 'rgb(253,93,124)', 'rgb(28,119,139)', 'rgb(182,231,235)', 'rgb(35,154,160)']

n_phase = len(df_usa_sport['Sport'])
plot_width = 200

# height of a section and difference between sections 
section_h = 100
section_d = 10

# multiplication factor to calculate the width of other sections
unit_width = plot_width / max(df_usa_sport['Medal'])

# width of each funnel section relative to the plot width
phase_w = [int(value * unit_width) for value in df_usa_sport['Medal']]

height = section_h * n_phase + section_d * (n_phase - 1)

# list containing all the plot shapes
shapes = []

# list containing the Y-axis location for each section's name and value text
label_y = []

for i in range(n_phase):
        if (i == n_phase-1):
                points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
        else:
                points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

        path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': colors[i],
                'line': {
                    'width': 1,
                    'color': colors[i]
                }
        }
        shapes.append(shape)
        
        # Y-axis location for this section's details (text)
        label_y.append(height - (section_h / 2))

        height = height - (section_h + section_d)
        
label_trace = go.Scatter(
    x=[-200]*n_phase,
    y=label_y,
    mode='text',
    text=df_usa_sport['Sport'],
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)
 
# For phase values
value_trace = go.Scatter(
    x=[-350]*n_phase,
    y=label_y,
    mode='text',
    text=df_usa_sport['Medal'],
    textfont=dict(
        color='rgb(200,200,200)',
        size=12
    )
)

data = [label_trace, value_trace]
 
layout = go.Layout(
    title="<b>Top 10 Sports in which USA is best</b>",
    titlefont=dict(
        size=12,
        color='rgb(203,203,203)'
    ),
    shapes=shapes,
    height=600,
    width=800,
    showlegend=False,
    paper_bgcolor='rgba(44,58,71,1)',
    plot_bgcolor='rgba(44,58,71,1)',
    xaxis=dict(
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False
    )
)
 
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[112]:


df_USA=df[(df['Team']=='United States')]
df_USA_medal=df_USA[df_USA['Medal']=='Gold']

medal_map = {'Gold':1}
df_USA_medal['Medal'] = df_USA_medal['Medal'].map(medal_map)

df_USA_sport=df_USA_medal.groupby(['Sport'],as_index=False)['Medal'].agg('sum')

df_USA_sport=df_USA_sport.sort_values(['Medal'],ascending=False)

df_USA_sport=df_USA_sport.head(10)

temp_series = df_USA_sport['Medal']
labels = df_USA_sport['Sport']
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Sports in which USA has won maximum Gold Medals',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="USA")


# In[113]:


df_china=df[(df['Team']=='China')]
df_china_medal=df_china[df_china['Medal']=='Gold']

medal_map = {'Gold':1}
df_china_medal['Medal'] = df_china_medal['Medal'].map(medal_map)

df_china_sport=df_china_medal.groupby(['Sport'],as_index=False)['Medal'].agg('sum')

df_china_sport=df_china_sport.sort_values(['Medal'],ascending=False)

df_china_sport=df_china_sport.head(10)

temp_series = df_china_sport['Medal']
labels = df_china_sport['Sport']
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Sports in which China has won maximum Gold Medals',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="china")


# In[114]:


df_medal=df.dropna(subset=['Medal'])

df_medal_male=df_medal[df_medal['Sex']=="M"]
df_medal_female=df_medal[df_medal['Sex']=="F"]


# In[115]:


df_medal_male_gold=df_medal_male[df_medal_male['Medal']=='Gold']

medal_map = {'Gold':1}
df_medal_male_gold['Medal'] = df_medal_male_gold['Medal'].map(medal_map)

df_medal_male_gold=df_medal_male_gold.groupby(['Sport'],as_index=False)['Medal'].agg('sum')

df_medal_female_gold=df_medal_female[df_medal_female['Medal']=='Gold']

df_medal_female_gold['Medal'] = df_medal_female_gold['Medal'].map(medal_map)

df_medal_female_gold=df_medal_female_gold.groupby(['Sport'],as_index=False)['Medal'].agg('sum')


# In[116]:


temp1 = df_medal_male_gold[['Sport', 'Medal']] 
temp2 = df_medal_female_gold[['Sport', 'Medal']] 
# temp1 = gun[['state', 'n_killed']].reset_index(drop=True).groupby('state').sum()
# temp2 = gun[['state', 'n_injured']].reset_index(drop=True).groupby('state').sum()
trace1 = go.Bar(
    x=temp1.Sport,
    y=temp1.Medal,
    name = 'Sports in which Males have won max. Gold Medals'
)
trace2 = go.Bar(
    x=temp2.Sport,
    y=temp2.Medal,
    name = 'Sports in which Females have won max. Gold Medals'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Sports in which Males have won max. Gold Medals', 'Sports in which Females have won max. Gold Medals'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout']['xaxis1'].update(title='Name of Sport')
fig['layout']['xaxis2'].update(title='Name of Sport')

fig['layout']['yaxis1'].update(title='Sports in which Males have won Gold Medals')
fig['layout']['yaxis2'].update(title='Sports in which Females have won Gold Medals')
                          
fig['layout'].update(height=500, width=1500, title='Sports in which Males and Females have won max. Gold Medals')
iplot(fig, filename='simple-subplot')


# In[117]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import folium
from folium import plugins
import geopandas as gpd
import branca

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from scipy.stats import ttest_ind

plt.rcParams["font.family"] = "monospace"
plt.rcParams['figure.dpi'] = 150
background_color='#F5F4EF'

# Print colored text 
# https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
# Includes other color options

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

class color_font:
    S = BOLD + OKBLUE +  UNDERLINE   #S = Start
    E = ENDC #E = End
    
print(color_font.S+"Datasets & Libraries loaded"+color_font.E)


# In[118]:


population = pd.read_csv('./population_by_country_2020.csv')
regions = pd.read_csv('./noc_regions.csv')

df = pd.read_csv('./athlete_events.csv')
df_21 = pd.read_csv('./Tokyo 2021 dataset.csv')
df_21_full = pd.read_csv('./Tokyo 2021 dataset.csv')


# In[119]:


fig, ax = plt.subplots(figsize=(4, 5), facecolor=background_color)

temp = df_21_full[:15].sort_values(by='Total')
my_range=range(1,len(df_21_full[:15]['Team/NOC'])+1)


ax.set_facecolor(background_color)

#B73832



plt.hlines(y=my_range, xmin=0, xmax=temp['Total'], color='gray')
plt.plot(temp['Total'], my_range, "o",markersize=10, color='#244747')
plt.plot(temp['Total'][2], my_range[10], "o",markersize=20,color='#B73832')


Xstart, Xend = ax.get_xlim()
Ystart, Yend = ax.get_ylim()

ax.tick_params(axis=u'both', which=u'both',length=0)
ax.set_xlabel("Total Medals",fontfamily='monospace',loc='left',color='gray')
ax.set_axisbelow(True)


for s in ['top','right','bottom','left']:
    ax.spines[s].set_visible(False)
    


ax.text(-90,Yend+2.3, 'Olympic Total Medals by Country: Tokyo 2021', fontsize=15,fontweight='bold',fontfamily='serif',color='#323232')
ax.text(-90,Yend+1.1, 'Japan hosted the games for the second time', fontsize=10,fontweight='bold',fontfamily='sansserif',color='#B73832')
#ax.text(-100,Yend+1, 'Not Hosting', fontsize=10,fontweight='bold',fontfamily='sansserif',color='#244747')



# Add titles and axis names
plt.yticks(my_range, temp['Team/NOC'])
plt.xlabel('')


ax.annotate(temp['Total'][2], xy=(54.86,10.95), va = 'center', ha='left',fontweight='light', fontfamily='monospace',fontsize=10, color='white',rotation=0)

# Show the plot
plt.show()


# In[120]:


def highlight(nation):
    if nation['Team/NOC'] == 'Japan':
        return ['background-color: #f3f2f1']*6
    else:
        return ['background-color: white']*6

df_21_full[['Rank','Team/NOC','Bronze Medal','Silver Medal','Gold Medal','Total']].iloc[:15].style.set_caption('Medals by Country: Summer Olympic Games sorted by Gold Medals [Top 15]').bar(subset=['Gold Medal'], color='#f0c05a').bar(subset=['Silver Medal'], color='Lightgray').bar(subset=['Bronze Medal'], color='#a97142').hide_index().apply(highlight, axis=1)


# In[124]:


# For geographic plotting
global_polygons = gpd.read_file('./athlete_events.csv')
global_polygons.to_file('global_polygons.geojson', driver = 'GeoJSON')

#global_polygons.plot(figsize=(10,5)) we now have a map of the globe

# Tabular
df = pd.merge(df,regions,left_on='NOC',right_on='NOC')
df = df.query('Season == "Summer"') # Only interested in Summer Olympics for this project


# In[ ]:


# For geographic plotting
import geopandas as gpd
from matplotlib import pyplot as plt
global_polygons = gpd.read_file(country_shapes)
global_polygons.to_file('global_polygons.geojson', driver = 'GeoJSON')

#global_polygons.plot(figsize=(10,5)) we now have a map of the globe

# Tabular
df = pd.merge(df,regions,left_on='NOC',right_on='NOC')
df = df.query('Season == "Summer"') # Only interested in Summer Olympics for this project


# In[ ]:


#Replacing the country name with common values
df.replace('USA', "United States of America", inplace = True)
df.replace('Tanzania', "United Republic of Tanzania", inplace = True)
df.replace('Democratic Republic of Congo', "Democratic Republic of the Congo", inplace = True)
df.replace('Congo', "Republic of the Congo", inplace = True)
df.replace('Lao', "Laos", inplace = True)
df.replace('Syrian Arab Republic', "Syria", inplace = True)
df.replace('Serbia', "Republic of Serbia", inplace = True)
df.replace('Czechia', "Czech Republic", inplace = True)
df.replace('UAE', "United Arab Emirates", inplace = True)
df.replace('UK', "United Kingdom", inplace = True)

population.replace('United States', "United States of America", inplace = True)
population.replace('Czech Republic (Czechia)', "Czech Republic", inplace = True)
population.replace('DR Congo', "Democratic Republic of the Congo", inplace = True)
population.replace('Serbia', "Republic of Serbia", inplace = True)
population.replace('Tanzania', "United Republic of Tanzania", inplace = True)

df_21_full.replace('Great Britain', "United Kingdom", inplace = True)
df_21_full.replace("People's Republic of China", "China", inplace = True)
df_21_full.replace("ROC", "Russia", inplace = True)


# In[ ]:


# Function to map country to city

def host_country(col):
    if col == "Rio de Janeiro":
        return "Brazil"
    elif col == "London":
        return "United Kingdom"
    elif col == "Beijing":
        return  "China"
    elif col == "Athina":
        return  "Greece"
    elif col == "Sydney" or col == "Melbourne":
        return  "Australia"
    elif col == "Atlanta" or col == "Los Angeles" or col == "St. Louis":
        return  "United States of America"
    elif col == "Barcelona":
        return  "Spain"
    elif col == "Seoul":
        return  "South Korea"
    elif col == "Moskva":
        return  "Russia"
    elif col == "Montreal":
        return  "Canada"
    elif col == "Munich" or col == "Berlin":
        return  "Germany"
    elif col == "Mexico City":
        return  "Mexico"
    elif col == "Tokyo":
        return  "Japan"
    elif col == "Roma":
        return  "Italy"
    elif col == "Paris":
        return  "France"
    elif col == "Helsinki":
        return  "Finland"
    elif col == "Amsterdam":
        return  "Netherlands"
    elif col == "Antwerpen":
        return  "Belgium"
    elif col == "Stockholm":
        return  "Sweden"
    else:
        return "Other"


# Applying this function

df['Host_Country'] = df['City'].apply(host_country)


# In[ ]:


df_new = df.groupby(['Year','Host_Country','region','Medal'])['Medal'].count().unstack().fillna(0).astype(int).reset_index()

df_new['Is_Host'] = np.where(df_new['Host_Country'] == df_new['region'],1,0)
df_new['Total Medals'] = df_new['Bronze'] + df_new['Silver'] + df_new['Gold']


# In[97]:


import pandas as pd 
import numpy as np 
import math
import glob
import os

# visualization
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


# In[98]:


df3 = pd.read_csv('./Tokyo Medals 2021.csv')
df3.head()


# In[99]:


df3.shape


# In[100]:


df3.columns


# In[101]:


df3.info()


# In[102]:


df3.isnull().sum()


# In[103]:


df3.describe()


# In[104]:


# Histogram
f, ax = plt.subplots(figsize=(20,5))
sns.distplot(df3['Total'])


# In[105]:


# KDE Plot 
f, ax = plt.subplots(figsize=(20,5))
sns.kdeplot(df3['Total'])


# In[106]:


df3['Country'].value_counts().index[:10]


# In[107]:


plt.figure(figsize=(10, 10))
heatmap = sns.heatmap(df3.corr(),annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# In[108]:


plt.figure(figsize=(20,10),dpi=300)
sns.scatterplot(x='Total',y='Country',hue='Country',data=df3[:30])


# In[109]:


plt.figure(figsize=(20,10))
sns.swarmplot(x='Total',y='Country',hue='Country',data=df3[:30])


# In[110]:


plt.figure(figsize=(20,10))
sns.swarmplot(x='Gold Medal',y='Country',hue='Country',data=df3[:30])


# In[111]:


plt.figure(figsize=(20,10))
sns.swarmplot(x='Silver Medal',y='Country',data=df3[:30],hue='Country')


# In[112]:


plt.figure(figsize=(20,10))
sns.swarmplot(x='Bronze Medal',y='Country',data=df3[:30],hue='Country')


# In[113]:


plt.figure(figsize=(20, 20))
plt.tight_layout()
sns.barplot(x='Total',y='Country',data=df3)
plt.title('All countries by medals')
plt.show()


# In[114]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.barplot(x='Gold Medal',y='Country',color = '#D1B000',data=df3[:15])
plt.title('Distribution of Gold Medals')
plt.show()


# In[115]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.barplot(x='Silver Medal',y='Country',color='#828A95',data=df3[:15])
plt.title('Distribution of Silver Medals')
plt.show()


# In[116]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.barplot(x='Bronze Medal',y='Country',color = '#914E24',data=df3[:20])
plt.title('Distribution of Bronze Medals')
plt.show()


# In[117]:


sns.lmplot(x="Gold Medal", y="Silver Medal", hue="Country", data=df3[:10])


# In[118]:


sns.pairplot(df3)


# In[119]:


df_india = df3.loc[df3['Country']=='India']
df_india


# In[120]:


df3['Country'].unique()


# In[121]:


no_of_countries = len(df3['Country'].unique())
no_of_countries


# In[122]:


df3


# In[123]:


top_15_countries = df3['Country'][:15]
top_15_total = df3['Total'][:15]


#add colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

figure = plt.figure(figsize=(15,10))
plt.pie(top_15_total, labels=top_15_countries,
        colors = colors, shadow = True, startangle = 90, autopct='%1.2f%%')

plt.title(' Pie Plot for Top 15 Countries - Rank by Total')
plt.tight_layout()
plt.show()


# In[124]:


top_15_countries = df3['Country'][:15]
top_15_total = df3['Total'][:15]

# set value to 0.1 if you wish to highlight particular country else assign 0 
explode = (0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0)

#add colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

figure = plt.figure(figsize=(15,10))
plt.pie(top_15_total, labels=top_15_countries,
        explode = explode, colors = colors, shadow = True, startangle = 90, autopct='%1.2f%%')

plt.title(' Pie Plot for Top 15 Countries - Rank by Total')
plt.tight_layout()
plt.show()


# In[125]:


top_15_countries = df3['Country'][:15]
top_15_total = df3['Total'][:15]


#add colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

figure = plt.figure(figsize=(15,10))
plt.pie(top_15_total, labels=top_15_countries,
        colors = colors, shadow = True, startangle = 90, autopct='%1.2f%%')

#draw circle
centre_circle = plt.Circle((0,0),0.75,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)


plt.title(' Donut Chart for Top 15 Countries - Rank by Total')
plt.tight_layout()
plt.show()


# In[126]:


## Donut chart - highlight certain countries 

top_15_countries = df3['Country'][:15]
top_15_total = df3['Total'][:15]


#add colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

explode = (0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0)

figure = plt.figure(figsize=(15,10))
plt.pie(top_15_total, labels=top_15_countries,
        colors = colors, shadow = True, pctdistance=0.85, startangle = 90, autopct='%1.2f%%',explode=explode)

#draw circle
centre_circle = plt.Circle((0,0),0.75,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)


plt.title(' Donut Chart for Top 15 Countries - Rank by Total')
plt.tight_layout()
plt.show()


# In[127]:


top_15_countries = df3['Country'][:15]
top_15_total = df3['Total'][:15]



#add colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

figure = plt.figure(figsize=(15,10))
plt.pie(top_15_total, labels=top_15_countries,
        colors = colors, shadow = True, startangle = 90, autopct='%1.2f%%')

plt.title(' Pie Plot for Top 15 Countries - Rank by Total')
plt.tight_layout()
plt.show()


# In[128]:


top_15 = df3[:15]
top_15


# In[129]:


fig,ax = plt.subplots(figsize = (20,10))
plt.bar(top_15['Country'].values, top_15['Gold Medal'].values, color = '#C49133', label = 'Gold',)
plt.bar(top_15['Country'].values, top_15['Silver Medal'].values, width=0.5,color = '#828A95', label = 'Silver')
plt.bar(top_15['Country'].values, top_15['Bronze Medal'].values, width=0.25, color = '#914E24', label = 'Bronze')

plt.title('Medals won by top 15 Countries at Olympics', fontweight = 'bold', fontsize=20)
plt.xlabel('Country Name', fontsize = 10, fontweight = 'bold')
plt.ylabel('No of Medals', fontsize = 10, fontweight = 'bold')
plt.legend(fontsize = 20)



ax.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(fontsize=14, rotation=45)
plt.tight_layout()
plt.show()


# In[130]:


# these two source have been so helpful in understanding how to plot side by side bar plot
# https://stackoverflow.com/questions/10369681/how-to-plot-bar-graphs-with-same-x-coordinates-side-by-side-dodged
# https://www.kaggle.com/sujan97/data-visualization-quick-guide
 

fig,ax = plt.subplots(figsize = (20,10))

top_15_country_names = top_15['Country'].values
top_15_gold_medals = top_15['Gold Medal'].values
top_15_silver_medals = top_15['Silver Medal'].values
top_15_bronze_medals = top_15['Bronze Medal'].values
indices = 15 #Numbers of pairs of bars you want
ind = np.arange(indices) #Position of bars on x-axis


width = 0.3 #width of bars
ax.bar(ind, top_15_gold_medals, data=top_15,width=width,color = '#C49133', label = 'Gold')
ax.bar(ind+width, top_15_silver_medals, data=top_15,width=width,color = '#828A95', label = 'Silver')
ax.bar(ind+2*width, top_15_bronze_medals, data=top_15,width=width, color = '#914E24', label = 'Bronze')



plt.title('Medals won by top 15 Countries at Olympics', fontweight = 'bold', fontsize=20)
plt.xlabel('Country Name', fontsize = 10, fontweight = 'bold')
plt.ylabel('No of Medals', fontsize = 10, fontweight = 'bold')
plt.legend(fontsize = 20)



ax.tick_params(axis='both', which='major', labelsize=15)
#plt.xticks(fontsize=14, rotation=45)
#ax.set_xticks(ind + width / 2)


plt.xticks(ind+width/2,top_15_country_names,fontsize=14,rotation=45)
plt.tight_layout()
plt.show()


# In[131]:


# Horizontal Stacked Barplot

fig,ax = plt.subplots(figsize = (20,10))
plt.barh(top_15['Country'].values, top_15['Gold Medal'].values, color = '#C49133', label = 'Gold')
plt.barh(top_15['Country'].values, top_15['Silver Medal'].values, color = '#828A95', label = 'Silver')
plt.barh(top_15['Country'].values, top_15['Bronze Medal'].values,  color = '#914E24', label = 'Bronze')

plt.title('Medals won by top 15 Countries at Olympics', fontweight = 'bold', fontsize=20)
plt.xlabel('No of Medals', fontsize = 10, fontweight = 'bold')
plt.ylabel('Country Name', fontsize = 10, fontweight = 'bold')
plt.legend(fontsize = 20)


ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.show()


# In[132]:


#https://www.pythonprogramming.in/plot-polar-graph-in-matplotlib.html
#https://www.kaggle.com/sujan97/data-visualization-quick-guide

fig,ax=plt.subplots(figsize=(20,10), 
                    subplot_kw=dict(polar=True))

x=top_15_countries
y1= top_15_gold_medals
y2= top_15_silver_medals
y3 = top_15_bronze_medals

theta = np.linspace(0, 2 * np.pi, len(top_15_countries))

# Arrange the grid 
lines, labels = plt.thetagrids(range(0, 360, int(360/len(top_15_countries))), (top_15_countries))

ax.bar(x,y1, label='Gold', color = '#C49133')
ax.bar(x,y2, label='Silver', color = '#828A95')
ax.bar(x,y3, label='Bronze',  color = '#914E24')




ax.legend()
plt.show()


# In[133]:


fig,ax = plt.subplots(figsize = (20,10))
plt.bar(top_15['Country'].values, top_15['Total'].values, width=0.25,color = '#33A6BD', label = 'Total Medals')

plt.title('Medals won by top 15 Countries at Olympics', fontweight = 'bold', fontsize=20)
plt.xlabel('No of Medals', fontsize = 10, fontweight = 'bold')
plt.ylabel('Country Name', fontsize = 10, fontweight = 'bold')
plt.legend(fontsize = 20)


ax.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(rotation=45)
plt.tight_layout()
plt.show()


# In[134]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('df3'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import folium
from folium import plugins
import geopandas as gpd
import branca

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from scipy.stats import ttest_ind

plt.rcParams["font.family"] = "monospace"
plt.rcParams['figure.dpi'] = 150
background_color='#F5F4EF'

# Print colored text 
# https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
# Includes other color options

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

class color_font:
    S = BOLD + OKBLUE +  UNDERLINE   #S = Start
    E = ENDC #E = End
    
print(color_font.S+"Datasets & Libraries loaded"+color_font.E)


# In[ ]:





# In[ ]:




