#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels as sm
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
sns.set()


historical_weather_data = pd.read_csv("/Users/colleenjung/Desktop/UChicago/Hackathon23/historical_weather.csv")


# In[5]:


data = historical_weather_data



# In[4]:


list(data.columns.values)
data.notna().sum()


# In[5]:


data=data.drop(['name','tempmin','tempmax','feelslikemax','feelslikemin','feelslike','dew','humidity','snow','snowdepth','sealevelpressure','solarradiation','solarenergy','uvindex','sunrise','sunset','moonphase','stations','icon','description'],axis=1)


# In[6]:


(data.columns.values)
data['preciptype'].fillna(0,inplace=True)
data['windgust'].fillna(0,inplace=True)
data['severerisk'].fillna(0,inplace=True)


# In[7]:


column_names = ['datetime','temp','precip','precipprob','preciptype','windspeed','cloudcover','visibility','conditions']



# In[16]:


data['datetime'] = pd.to_datetime(data['datetime'])

october_data = data[(data['datetime'].dt.month == 10) & 
                    (data['datetime'].dt.year >= 2010) & 
                    (data['datetime'].dt.year <= 2023)]
october_data.to_csv('october_dates_2010_2023.csv', index=False)
october_data.set_index('datetime', inplace=True)


# In[17]:


october_data = october_data.copy()


# In[19]:


october_data['temp'].resample('A').mean().plot()


# In[21]:


october_data = october_data.reset_index()

october_data['datetime'] = october_data['datetime'].astype('str')
october_data['precipprob'] = october_data['precipprob'].astype('str')
october_data['precipcover'] = october_data['precipcover'].astype('float')
october_data['preciptype'] = october_data['preciptype'].map({'rain': 1, 'rain,snow': 2, 'snow': 3})
october_data['conditions'] = october_data['conditions'].astype('str')


# In[22]:


october_data.describe()


# In[23]:


print(october_data.columns)


# In[28]:


#Rolling Average: Moving averages. 

rolling_temp = october_data['temp'].rolling(window=7).mean()
plt.figure(figsize=(14, 7))
plt.plot(october_data['datetime'], october_data['temp'], label='Original Temp')
plt.plot(october_data['datetime'], rolling_temp, label='7-day Rolling Average', color='red')
plt.title('Temperature over Time with Rolling Average')
plt.legend()
plt.show()

#분석의 대한 설명 
# 저 아래 보면, 10월의 daily temp 랑 7일동안의 avg temp 를 그래프로 보여주고있어. 
# 파랑이 original temp 고 빨강은 7 day avg temp (rolling avg)
# 저거 보면, 7일 평균을 사용해서, 단기적인 temp의 변동(transition)을 줄여서, 
# overall trend 를 더 clear 하게 볼수 있어. 
# 빨강 보면, 중요한 변동 선을 보여주는것, 파랑은 일별 온도의 직접적인 변화를 보여줘 
# unique 한 패턴을 탐색 쌉 가능. 


# In[29]:


#Vector Auto Regression Model : **This is the time series analysis**

fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.set_xlabel('Datetime')
ax1.set_ylabel('Temperature', color='tab:blue')
ax1.plot(october_data['datetime'], october_data['temp'], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Precipitation', color='tab:green')
ax2.plot(october_data['datetime'], october_data['precip'], color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

plt.title('Temperature and Precipitation over Time')
plt.show()

#분석의 대한 설명: 
# 이 그래프를 보면, temp 랑 precip 이라는 두 정보를 , 'time' 에 따라 보여주는 그래프야. 
# temp 가 파랑, precip 이 초록이야. 이것을 통해 날짜별로 temp 랑 precip 을 함께 확인할수있지.
# 이 관계를 보면, 특정 날짜에 온도가 높다, 낮다, 그걸 기준으로 감수량이 어떻게 변화되는지 확인가능하고
# 변동성을 확인할수 있고 (특정 시간동안 얼마나 변동했는지..)
# 패턴인식. 즉, 특정 기간동안 temp 랑 precip 의 특이한 패턴이 있다, 그러면 그 기간동에 관찰하면돼.


# In[30]:


#PCA : Principle Component Analysis . 
# you basically simplify many variables of the datasets and put them all together in one dimension.
# very useful for visualization. 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

numeric_data = october_data[['temp', 'precip', 'windgust', 'windspeed', 'winddir', 'cloudcover', 'visibility']]
scaled_data = StandardScaler().fit_transform(numeric_data)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result')
plt.show()

#분석의 대한 설명 : 
#데이처의 크기나 차원을 확 줄여서, 여러정보를 간단한 형태로 만드는거야. 예를 들어, 보이는 점들의 분포를 그래프로 보여주는거지. 
# 저 점들을 보면, 점들이 어디에 모여있는지 볼수 있잖아 그곳에 많은 점들이 있으면, 그곳에 특별한 패턴이나 정보가 있다는 뜻이야. 
# 또, 일반적인 패턴에서 벗어난 점들을 보며, 의미부여를 할수도 있겠지. 그건 분석가 맘이고, 상황에 따라 다르게 선택하는것. 
#중요한건, 분산에 관련된건데, PCA 는 중요한 정보의 양을 나타내기 때문에 더 큰 분산은 더 많은 정보를 포함하고 있다니까, 거기에 집중하면 돼. 
# 스케일링을 보면, 데이터의 크기를 조절하는것. 그러니까, 모든 정보를 공정하게 비교할수 있을거라는 점 알아야해. 


# In[42]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Cluster Analysis
features = ['temp', 'precip', 'windgust', 'windspeed', 'cloudcover', 'visibility']
X = october_data[features]

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
october_data['cluster'] = kmeans.labels_

sns.pairplot(october_data, hue='cluster', vars=features)
plt.show()
#분석의 대한 설명 : 
# 유사한 패턴이 보이는 데이터 points 를 그룹화 하는거야. 데이터를 묶어서 비슷한 특성들끼리.
# kmeans 클러스터링이 중요한데, 데이터 포인ㄷ트를 여러 그룹으로 만들어. 코드 를보면, 총 4개의 클러스터를 만들었지.
# 그담에 시각화했음. 알아서 보시길! 

#Time series Decomposition
result = seasonal_decompose(ts_data, model='additive', period=24)
result.plot()
plt.show()

#분석의 대한 설명: 
# 먼저 데이터를 선택해서, precip data 를 ts_data로. 
# seasonal decompose 를 사용해서 데이터를 time series 로 분해했으. 
# 시각화 함. 
# 아래 보면 trend, seasonal, resid(residual = left over data) 로 나눴음. 
# trend 는 일반적 변화를 보여줌
# seasonal 은 주기적 변화 보여줌
# left over 는 예측할수 없는 나머지의 데이터니까, 변동성의 정도를 설명해줌. 


# In[43]:


print(october_data.info())
october_data['preciptype'].fillna(0, inplace=True)


# In[44]:


#october_data.to_csv('/Users/hjk2160@columbia.edu/Desktop/october_data.csv', index=False)


# In[ ]:




