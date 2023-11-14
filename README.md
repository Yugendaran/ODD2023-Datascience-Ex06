# EX NO : 06 Feature Transformation

## DATE : 

## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM:

### STEP 1:
Read the given Data

### STEP 2:
Clean the Data Set using Data Cleaning Process

### STEP 3:
Apply Feature Transformation techniques to all the features of the data set

### STEP 4:
Print the transformed features

## CODE:
DEVELOPED BY: YUGENDARAN G

REGISTER NO: 212221220063
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
## OUTPUT:
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex06/assets/128135616/0f947eec-27d3-4fee-97c1-efe9e211fe18)

![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex06/assets/128135616/4c4713a9-7492-4145-a4da-a2f72baec6e7)

![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex06/assets/128135616/bbf930a7-dea8-466c-982f-a6724b7995c1)

![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex06/assets/128135616/ad582034-9b90-455b-b63b-6a8362f83996)

![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex06/assets/128135616/04b9110b-83fb-414e-970e-93adb094d9c6)

![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex06/assets/128135616/d422bc21-3c14-4716-ba54-99d16ec6ba08)

![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex06/assets/128135616/ccc36005-3609-4e15-973d-e2a3195ffc09)

![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex06/assets/128135616/20db8441-b1b6-4843-9425-044e48e17244)

![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex06/assets/128135616/55c225e1-ce48-4bd4-b01b-c58eb838075a)

![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex06/assets/128135616/4d91db04-a632-4190-bc84-fd4e84dfd2fe)

## RESULT:
Thus feature transformation is done for the given dataset.










