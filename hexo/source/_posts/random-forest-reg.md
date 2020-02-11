---
title: 随机森林简易流程
date: 2019-06-26 21:21:36
tags: 机器学习
---


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

---
```

 >   C:\Users\Yaotc\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    
---

```python
df = pd.read_excel('./dataset_v3.0.xlsx')
df.keys()
```




 >   Index(['Sample', 'Gra_diameter_nm', 'oxidation_state', 'surface_modif',
           'cell_morph', 'cell_source', 'cell_line', 'organ_source',
           'detection_method', 'Exposure-time-hrs', 'Cell-viability-percent',
           'exposure_dose', 'Reference', '文献号'],
 >         dtype='object')


---

```python
corr_data = df.iloc[:,list(range(1,10))+[11,10] ]
corr_data_ans = corr_data.corr()
corr_data[:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gra_diameter_nm</th>
      <th>oxidation_state</th>
      <th>surface_modif</th>
      <th>cell_morph</th>
      <th>cell_source</th>
      <th>cell_line</th>
      <th>organ_source</th>
      <th>detection_method</th>
      <th>Exposure-time-hrs</th>
      <th>exposure_dose</th>
      <th>Cell-viability-percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>450.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>99.691404</td>
    </tr>
    <tr>
      <th>1</th>
      <td>450.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>94.243601</td>
    </tr>
  </tbody>
</table>
</div>


---

```python
plt.figure(figsize=(10,10))
sns.heatmap(corr_data_ans, annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
plt.savefig('heatmap.jpg')
plt.show()
```


![png](/images/RandomForestRegressor_heatmap.png)

---

```python
RANDOM_SEED = 666
x_data = df.iloc[:,list(range(1,10))+[11] ]
y_data = df['Cell-viability-percent']

x_data = np.array(x_data)
y_data = np.array(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=RANDOM_SEED, shuffle=True)

print(f"Number of train: {len(x_train)}\nNumber of test: {len(x_test)}")
```

>    Number of train: 887
    Number of test: 99
    
---

```python
# TODO: k-fold 
# kf = KFold(n_splits=10,random_state=RANDOM_SEED,shuffle=False)

# for train_index , test_index in kf.split(x_data):
# #     print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_test = x_data[train_index], x_data[test_index]
#     y_train, y_test = y_data[train_index], y_data[test_index]
```
---

```python
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1,1))
y_test = ss_y.transform(y_test.reshape(-1,1))

# Random Forest Regressor
rfr = RandomForestRegressor(random_state=RANDOM_SEED)
# train
rfr.fit(x_train, y_train)
# predict
rfr_y_predict = rfr.predict(x_test)

print("Score:", rfr.score(x_test, y_test))
print("R_squared：", r2_score(y_test, rfr_y_predict))
print("MSE:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print("MAE:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))

```

 >   Score: 0.7747153143286902
    R_squared： 0.7747153143286903
    MSE: 186.5440944695794
    MAE: 9.049509436791483
    
---

```python
plt.figure(figsize=(10,10))
plt.xlim(0,len(y_test))
plt.ylim(min(rfr_y_predict.min(),y_test.min())-2,max(rfr_y_predict.max()+2,y_test.max()))
# plt.scatter
plt.scatter(list(range(len(y_test))),y_test, color='c', marker = 'x', label='ori', s=15)
plt.scatter(list(range(len(rfr_y_predict))),rfr_y_predict, color='m', marker = 'o', label='pre', s=15)
plt.legend(loc = 'upper left')
# plt.grid(True)
plt.text(len(y_test)//2,max(rfr_y_predict.max(),y_test.max()),"R-squard={}".format(r2_score(y_test, rfr_y_predict)))
plt.savefig("dots.jpg")
plt.show()
```


![png](/images/RandomForestRegressor_dots.png)

---

```python
plt.figure(figsize=(10,10))
plt.xlim(0,len(y_test))
plt.ylim(min(rfr_y_predict.min(),y_test.min())-2,max(rfr_y_predict.max(),y_test.max())+2)
# plt.scatter
plt.plot(list(range(len(y_test))),y_test, color='c', marker = 'x', label='ori', )
plt.plot(list(range(len(rfr_y_predict))),rfr_y_predict, color='m', marker = 'o', label='pre', )
plt.legend(loc = 'upper left')
# plt.grid(True)
plt.text(len(y_test)//2,max(rfr_y_predict.max(),y_test.max()),"R-squard={}".format(r2_score(y_test, rfr_y_predict)))
plt.savefig("lines.jpg")
plt.show()
```


![png](/images/RandomForestRegressor_line.png)

