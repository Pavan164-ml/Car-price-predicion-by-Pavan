#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


df = pd.read_csv('CarData.csv.csv')


# In[3]:


final_data = df[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[4]:


final_data['current_year'] = 2020


# In[5]:


final_data['Number_of_years'] = final_data['current_year'] - final_data['Year']


# In[6]:


final_data.drop(labels = ['Year','current_year'],axis=1,inplace = True)


# In[7]:


final_data = pd.get_dummies(final_data,drop_first = True)


# In[8]:


X = final_data.iloc[:,1:]
y = final_data.iloc[:,0]


# In[9]:


# Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[12]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV
#Randomized Search C V     ## HyperParameters


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[13]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[14]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[15]:


rf_random.fit(X_train,y_train)


# In[18]:


predictions = rf_random.predict(X_test)


# In[20]:


import pickle
# Open a file , where you want to store the data
file = open('random_forest_regression_model.pkl','wb')
# dump the information to that file
pickle.dump(rf_random,file)


# In[ ]:




