#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


train=pd.read_csv('/kaggle/input/janatahack-machine-learning-for-banking/train_fNxu4vz.csv')
test=pd.read_csv('/kaggle/input/janatahack-machine-learning-for-banking/test_fjtUOL8.csv')
sub=pd.read_csv('/kaggle/input/janatahack-machine-learning-for-banking/sample_submission_HSqiq1Q.csv')


# In[3]:


train.Loan_Amount_Requested = pd.Series([int("".join(i.split(","))) for i in train.Loan_Amount_Requested])


# In[4]:


test.Loan_Amount_Requested = pd.Series([int("".join(i.split(","))) for i in test.Loan_Amount_Requested])


# In[5]:


train.fillna(-999,inplace=True)
test.fillna(-999,inplace=True)


# In[6]:


import seaborn as sns


# In[7]:


# Remove column name 'A' 
train=train.drop(['Loan_ID'], axis = 1)


# In[8]:


# Remove column name 'A' 
test1=test.drop(['Loan_ID'], axis = 1)


# In[9]:


from catboost import Pool, CatBoostClassifier, cv, CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[13]:


#Separating label data for training
x = train.drop(['Interest_Rate'],axis=1)
y = train['Interest_Rate']


# Label encoding for all categorical varibles which are predictors

# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


label_encoding =LabelEncoder()


# In[14]:


label_encoding = label_encoding.fit(x['Length_Employed'].astype(str))


# In[15]:


label_encoding1 = LabelEncoder().fit(x['Home_Owner'].astype(str))
label_encoding2 = LabelEncoder().fit(x['Income_Verified'].astype(str))
label_encoding3 = LabelEncoder().fit(x['Purpose_Of_Loan'].astype(str))
label_encoding4 = LabelEncoder().fit(x['Gender'].astype(str))


# In[16]:


label_encoding5 = LabelEncoder().fit(test1['Home_Owner'].astype(str))
label_encoding6 = LabelEncoder().fit(test1['Income_Verified'].astype(str))
label_encoding7 = LabelEncoder().fit(test1['Purpose_Of_Loan'].astype(str))
label_encoding8 = LabelEncoder().fit(test1['Gender'].astype(str))
label_encoding9 = LabelEncoder().fit(test1['Length_Employed'].astype(str))


# In[17]:


x['Length_Employed'] = label_encoding.transform(x['Length_Employed'].astype(str))
x['Home_Owner'] = label_encoding1.transform(x['Home_Owner'].astype(str))
x['Income_Verified'] = label_encoding2.transform(x['Income_Verified'].astype(str))
x['Purpose_Of_Loan'] = label_encoding3.transform(x['Purpose_Of_Loan'].astype(str))
x['Gender'] = label_encoding4.transform(x['Gender'].astype(str))


# In[18]:


test1['Length_Employed'] = label_encoding9.transform(test1['Length_Employed'].astype(str))
test1['Home_Owner'] = label_encoding5.transform(test1['Home_Owner'].astype(str))
test1['Income_Verified'] = label_encoding6.transform(test1['Income_Verified'].astype(str))
test1['Purpose_Of_Loan'] = label_encoding7.transform(test1['Purpose_Of_Loan'].astype(str))
test1['Gender'] = label_encoding8.transform(test1['Gender'].astype(str))


# In[19]:


# dropping gender column because it is having less importance score
x1=x.drop(['Gender'],axis=1)


# In[20]:


# dropping gender column because it is having less importance score
test11=test1.drop(['Gender'],axis=1)


# Hyperparameter tuning using Ranomsearchcv for Xgbclassifier

# In[21]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[ ]:


n_estimators = [50,70,100,150,200,250, 300, 500, 800, 1200]
max_depth = [-1,1,2,3,4,5, 8,10, 15,20, 25, 30]
learning_rate=[0.01,0.05,0.1,0.2,0.3]
reg_lambda=[0.0,0.1,0.2,0.3,0.4,0.6,1]
reg_alpha=[0.0,0.1,0.2,0.3,0.4,0.6,1]
colsample_bytree=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
subsample=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
min_child_weight=[0.001,0.1,0.2,1]
gamma=[0.0,0.01,0.05,0.1,0.2,0.3]
random_state=[1,2,5,10,100,500,1000,2000]



hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,learning_rate=learning_rate,
             reg_lambda=reg_lambda,reg_alpha=reg_alpha,colsample_bytree=colsample_bytree,
             subsample=subsample,min_child_weight=min_child_weight,
             gamma=gamma,random_state=random_state)

gs = RandomizedSearchCV(
    estimator=xg, param_distributions=hyperF, 
    n_iter=5,
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)
bestF = gs.fit(x1, y)


# In[23]:


#make the x for train and test (also called validation data) 
xtrain,xtest,ytrain,ytest = train_test_split(x1,y,train_size=0.8,random_state=1)


# In[ ]:


from xgboost import XGBClassifier
xg=XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0.3, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.05, max_delta_step=0, max_depth=5,
              min_child_weight=0.001, monotone_constraints=None,
              n_estimators=1200, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=500, reg_alpha=0.4,
              reg_lambda=0.3, scale_pos_weight=None, subsample=0.5,
              tree_method=None, validate_parameters=False, verbosity=None)


# In[27]:


# i trained over whole data, it is given best result on public board using Xgbclassifier
xg.fit(x1,y)


# In[24]:


# Catboostclassifier
model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)


# In[25]:


#now just to make the model to fit the data
model.fit(xtrain,ytrain,eval_set=(xtest,ytest))


# In[28]:


# accuracy score for xgbclassifier
print('the test accuracy is :{:.6f}'.format(accuracy_score(ytest,xg.predict(xtest))))


# In[ ]:


import lightgbm as lgb
lgbm=lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8,
               importance_type='split', learning_rate=0.2, max_depth=15,
               min_child_samples=30, min_child_weight=0.001, min_split_gain=0.2,
               n_estimators=800, n_jobs=-1, num_leaves=31, objective=None,
               random_state=2, reg_alpha=0.4, reg_lambda=0.2, silent=True,
               subsample=0.3, subsample_for_bin=200000, subsample_freq=0)


# In[29]:


pred = xg.predict(test11)


# In[30]:


results=pd.DataFrame(pred,columns=['Interest_Rate'])


# In[31]:


xg.feature_importances_


# In[32]:


id1=pd.DataFrame(test.Loan_ID,columns=['Loan_ID'])


# In[33]:


final=pd.concat([id1,results],axis=1)


# In[ ]:


final.head()


# In[ ]:


final.to_csv('/kaggle/working/xggtuned5.csv',index=False)

