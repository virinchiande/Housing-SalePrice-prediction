
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd

data = pd.read_csv("C:\\Users\\virin\\OneDrive\\Documents\\train.csv")
data.head(5)


# In[53]:


data.info()
test1 = data.loc[:,['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1',
'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']]
test1.head(5)


# In[54]:


from pandas.tools.plotting import scatter_matrix

attributes = ['SalePrice','MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1',
'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']
scatter_matrix(data[attributes], figsize = (14,10))
plt.show()


# In[55]:


plt.scatter(x = data['SalePrice'], y = data['LotArea'], alpha = 0.5)
plt.show()


# In[56]:


plt.scatter(x = data['SalePrice'], y = data['GrLivArea'], alpha = 0.5)
plt.show()


# In[57]:


data['SalePrice'].describe()


# In[58]:


import matplotlib.pyplot as plt
plt.plot(data['SalePrice'])
plt.show()


# In[59]:


import seaborn as sns
#Right skewed data
#mean<median
sns.distplot(data['SalePrice']);
plt.show()


# In[60]:


#correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()


# In[61]:


#the highly correlated variables are selected
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data[cols], size = 2.5)
plt.show();


# In[62]:


#checking inter correlation
sns.set()
data_inter = data.loc[:,['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
sns.pairplot(data_inter, size = 2.5)
plt.show();


# In[63]:


#no inter correlation between the variables is observed
corrmat = data_inter.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()


# In[70]:


#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[71]:


#dealing with missing data
data = data.drop((missing_data[missing_data['Total'] > 1]).index,1)
data = data.drop(data.loc[data['Electrical'].isnull()].index)
data.isnull().sum().max() #just checking that there's no missing data missing...


# In[77]:


#histogram and normal probability plot
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
sns.distplot(data['SalePrice'], fit = norm);
fig = plt.figure()
res = stats.probplot(data['SalePrice'], plot=plt)
plt.show()


# In[83]:


#applying log transformation
data['SalePrice'] = np.log(data['SalePrice'])


# In[84]:


sns.distplot(data['SalePrice'], fit = norm);
fig = plt.figure()
res = stats.probplot(data['SalePrice'], plot=plt)
plt.show()


# In[87]:


#histogram and normal probability plot
sns.distplot(data['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['GrLivArea'], plot=plt)
plt.show()


# In[89]:


#data transformation
data['GrLivArea'] = np.log(data['GrLivArea'])

#transformed histogram and normal probability plot
sns.distplot(data['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['GrLivArea'], plot=plt)
plt.show()


# In[91]:


#histogram and normal probability plot
sns.distplot(data['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['TotalBsmtSF'], plot=plt)
plt.show()


# In[82]:


data['HasBsmt'] = pd.Series(len(data['TotalBsmtSF']), index=df_train.index)
data['HasBsmt'] = 0 
data.loc[data['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[92]:


#transform data
data.loc[data['HasBsmt']==1,'TotalBsmtSF'] = np.log(data['TotalBsmtSF'])


# In[94]:


#histogram and normal probability plot
sns.distplot(data[data['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(data[data['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.show()


# In[95]:


#convert categorical variable into dummy
data = pd.get_dummies(data)
data.head(10)


# In[116]:


import numpy as np
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
clf = RandomForestRegressor(random_state =42, max_features = 'auto')
param_grid = {'max_depth' : [10,20,30,40,50],
              'n_estimators' :[10,100,200,300]}
validator = GridSearchCV(clf, param_grid= param_grid) 
validator.fit(x_train,y_train)
for i in max_depth:
    print(validator.best_score_)

print(validator.best_estimator_.n_estimators)
print(validator.best_estimator_.max_depth)
print(validator.best_estimator_.max_features)


# In[125]:


from sklearn import model_selection
for cv in np.arange(2, 12, 2):
    GS = model_selection.GridSearchCV(
        cv=cv, estimator=RandomForestRegressor(random_state=42),
        param_grid={'max_depth' : [10,20,30,40,50],
                    'n_estimators' :[10,100,200,300]}
        )
    GS.fit(x_train, y_train)
    print(cv, GS.best_score_,GS.best_estimator_.max_depth,GS.best_estimator_.n_estimators)


# In[132]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

y_train = (data["SalePrice"])
x_train = data.loc[:,['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
x = x_train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_train = pd.DataFrame(x_scaled)

tree_reg = RandomForestRegressor(max_depth = 30, random_state=42, n_estimators = 300)
tree_reg.fit(x_train,y_train)
tree_pred = tree_reg.predict(x_train)
tree_pred = pd.DataFrame(tree_pred)
mat = pd.concat([tree_pred,y_train], axis =1)
print(mat)
dec_mse = mean_squared_error(tree_pred, y_train)
rmse = np.sqrt(dec_mse)
rmse 


# In[232]:


#testing on test dataset
df_test = pd.read_csv("C:\\Users\\virin\\OneDrive\\Documents\\test.csv")
df_test.head(5)


# In[233]:


#missing data
total = df_test.isnull().sum().sort_values(ascending=False)
percent = ((df_test.isnull().sum()/df_test.isnull().count())*100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)


# In[234]:


#dealing with missing data
df_test = df_test.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_test.head(10)
df_test.columns.get_loc("TotalBsmtSF")


# In[235]:


from sklearn.preprocessing import Imputer
import numpy as np

imputer = Imputer(strategy = "median")
x =df_test.iloc[:, np.r_[43,27]]


df_test= df_test.fillna((x.median()), inplace=True)

#df_test = df_test.drop(df_test.loc[df_test['GarageCars'].isnull()].index)
#df_test = df_test.drop(df_test.loc[df_test['TotalBsmtSF'].isnull()].index)
df_test.isnull().sum().max() #just checking that there's no missing data missing...


# In[236]:


#applying log transformation
df_test['GrLivArea'] = np.log(df_test['GrLivArea'])


# In[237]:


df_test['HasBsmt'] = pd.Series(len(df_test['TotalBsmtSF']), index=df_test.index)
df_test['HasBsmt'] = 0 
df_test.loc[df_test['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[238]:


#transform data
df_test.loc[df_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_test['TotalBsmtSF'])


# In[239]:


#convert categorical variable into dummy
df_test = pd.get_dummies(df_test)
df_test.head(10)


# In[240]:


import math
df_test1 = df_test.loc[:,['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
tree_pred = pd.DataFrame(tree_reg.predict(df_test1), columns = ["SalePrice"])
#tree_pred = list(tree_pred)
tree_pred = np.e**(tree_pred)
tree_pred.info()


# In[241]:


test_data_id = pd.DataFrame(df_test["Id"])
test_data_id.head(5)


# In[242]:


output_data=pd.concat([test_data_id, tree_pred], axis = 1)


# In[244]:


output_data.to_csv('output_data_1.csv', index = False)

