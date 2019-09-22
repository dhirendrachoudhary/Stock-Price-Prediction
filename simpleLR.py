import pandas as pd
import math,quandl,datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = "z5az8TajL8x-LxSndnz9"
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High']- df['Adj. Close']) / df['Adj. Close']*100.00
df['PCT_change'] = (df['Adj. Close']- df['Adj. Open']) / df['Adj. Open']*100.00

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forcast_col  = 'Adj. Close'
df.fillna(-99999,inplace = True)

forcast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forcast_col].shift(-forcast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forcast_out]
X_lately = X[-forcast_out:]

df.dropna(inplace = True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

clf = LinearRegression(n_jobs=10)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
forcast_set = clf.predict(X_lately)
df['Forcast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+= one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forcast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()