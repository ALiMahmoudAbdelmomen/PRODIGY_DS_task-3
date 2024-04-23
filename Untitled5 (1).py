#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder , StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
 


# In[2]:


Train=pd.read_csv(r"C:\Users\ali\Downloads\playground-series-s4e2\train.csv")


# # Explore data

# In[3]:


#Train.head()
#Train.tail()
Train.sample(10)


# In[4]:


Train.shape


# In[5]:


Train.columns


# In[6]:


Train.info()


# In[7]:


Train.isnull().sum()


# In[8]:


Train.duplicated().sum()


# In[9]:


Train.describe().round(2)


# # Transform Data

# In[10]:


la=LabelEncoder()


# In[11]:


obj=Train.select_dtypes(include='object')
non_obj=Train.select_dtypes(exclude='object')


# In[12]:


for i in range(0,obj.shape[1]):
    obj.iloc[:,i]=la.fit_transform(obj.iloc[:,i])


# In[13]:


obj=obj.astype("int")


# In[14]:


data=pd.concat([obj,non_obj],axis=1)


# In[15]:


test=pd.read_csv(r"C:\Users\ali\Downloads\playground-series-s4e2\test.csv")


# In[16]:


obj=test.select_dtypes(include='object')
non_obj=test.select_dtypes(exclude='object')
for i in range(0,obj.shape[1]):
    obj.iloc[:,i]=la.fit_transform(obj.iloc[:,i])
obj=obj.astype("int")    
test=pd.concat([obj,non_obj],axis=1)


# In[17]:


test.head()


# In[18]:


sc=StandardScaler()


# In[19]:


scal1=data[['Age']]
scal2=data[['Weight']]
data['Age']=sc.fit_transform(scal1)
data['Weight']=sc.fit_transform(scal2)


# In[20]:


scal1=test[['Age']]
scal2=test[['Weight']]
test['Age']=sc.fit_transform(scal1)
test['Weight']=sc.fit_transform(scal2)


# # create model

# In[21]:


x=data.drop(['NObeyesdad','id'],axis=1)
y=data['NObeyesdad']


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=21)


# In[23]:


model1=LogisticRegression()
model2=RandomForestClassifier()
model3=GaussianNB()
model4=SVC()
model5=XGBClassifier()
model6=GradientBoostingClassifier()


# In[24]:


def pred(model):
    model.fit(x_train,y_train)
    pre=model.predict(x_test)
    print(classification_report(pre,y_test))
    


# In[25]:


pred(model1)


# In[26]:


pred(model2)


# In[27]:


pred(model3)


# In[28]:


pred(model4)


# In[29]:


pred(model5)


# In[30]:


pred(model6)


# In[31]:


prim_grid={'n_estimators':[100,200,300],
         'learning_rate':[0.1,0.01,0.001],
          'max_depth':[3,5,7]}
scorer="accuracy"


# In[32]:


m5=GridSearchCV(model5,prim_grid,scoring=scorer, n_jobs=-1)
m5.fit(x_train,y_train)
print(m5.best_params_)
print(m5.best_score_)


# In[33]:


testx=test.drop('id',axis=1)


# In[34]:


prex=model5.predict(testx)


# In[35]:


submission=pd.DataFrame({"id":test['id'],"NObeyesdad":prex})


# In[36]:


#submission['NObeyesdad']=la.inverse_transform(prex)


# In[37]:


submission.to_csv("submission.csv",index=False) 


# In[38]:


submission


# In[ ]:




