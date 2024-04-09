#!/usr/bin/env python
# coding: utf-8

# In[273]:


#downloading dataset, importing libraries
import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier





from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.decomposition import PCA


# In[274]:


#reading the dataset, deleting missing value entries
df = pd.read_csv('data-breast cancer.csv')



df.head()



# In[275]:


#dropping columns with very low correlation and null values
df.drop(df.columns[[0,4,7,8,9,10,11,13,32]], axis =1, inplace = True)
df.columns


# In[276]:


#encoding diagnosis column as it contains string values M for Malignant and B for Benign
enc = OrdinalEncoder()


df[['diagnosis']] = enc.fit_transform(df[['diagnosis']])


# In[277]:


#visualization: correlation heatmap
plt.figure(figsize =(20,10))
sns.heatmap(df.corr(), annot = True)


# In[278]:


#selecting features based on high correlation

# radius mean, area_mean, radius_worst, perimeter_worst, area_worst, concavity worst, concavepoints_worst
selected_features = df[['radius_mean', 'area_mean', 'radius_worst', 'perimeter_worst', 'area_worst', 'concavity_worst', 'concave points_worst' ]]
display(selected_features)


# In[279]:


#selecting and scaling X variable (selected features)
scaler = StandardScaler()

X = scaler.fit_transform(selected_features)


# In[280]:


df['diagnosis'].value_counts().plot(kind = 'pie')
#Blue is Benign, #M is malignant


# In[281]:


#barplots to show individual correlation of selected features with diagnosis (Malignant and Benign)
fig, ax = plt.subplots(figsize = (15,30),nrows = 3, ncols = 3)
ax = ax.flatten()
for i, col in enumerate(selected_features):
    sns.barplot(x = 'diagnosis', y = col, data = df, ax = ax[i]);


# In[282]:


selected_features


# In[283]:


# defining y variable

y = df['diagnosis'].values


# In[284]:


#creating a split (train/test)
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=45)


#creating a split (train/val)

X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=45)

# now the train/validate/test split will be 80%/10%/10%


# In[285]:


#training random forest model for testing
scores = []
    
for k in range(1,20):
    model = RandomForestClassifier(random_state = 0, criterion = "entropy" , n_estimators = k)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))


# In[286]:


scores


# In[301]:


#plotting accuracy percentage vs n_estimators (testing)(Random Forest Model)


plt.plot(range(1,20), scores)
plt.title("Accuracy percent vs: n_estimators value, optimal value of k")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy Rate")

plt.show()


# In[288]:


#training KNeighborsClassifier for testing
acc_score = []

for k in range (1,30):
    knn = KNeighborsClassifier(n_neighbors = k)                              
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)
    acc_score.append(accuracy_score(y_test, y_test_pred))
    
   
    


# In[289]:


acc_score


# In[290]:


#plotting accuracy percentage vs n_neighbors (testing)(KNeighborsClassifier)


plt.plot(range(1,30), acc_score)
plt.title("Accuracy percent vs: n_neighbors value; optimal value of k")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy Rate")

plt.show()


# In[291]:


#training SVC (support vector classifier) for testing
scoreSVC = []

svm = SVC(kernel = 'rbf', gamma = 'auto', C = 0.2, random_state = 0).fit(X_train, y_train)
y_pred = svm.predict(X_test)
    
scoreSVC = svm.score(X_test,y_test)
con = confusion_matrix(y_test, y_pred)
print ("confusion matrix:\n",con)
cmval = confusion_matrix(y_val, y_pred)
cmval_display = ConfusionMatrixDisplay(cmval).plot()
    


# In[292]:


scoreSVC


# In[293]:


#conclusion
print ("Based on the above 3 models, KNeighbors classifier gives higher accuracy scores closer to 94% followed by Random Forest model")


# In[294]:


#Principal Component Analysis (PCA) for dimensionality reduction
#step 1: dropping diagnosis column to prevent double counting 
df = df.drop(columns = ['diagnosis'])


# In[295]:


#Principal Component Analysis (PCA) for dimensionality reduction
#step 2: scaling 
scaled = StandardScaler()
scaled.fit(df)


# In[296]:


#Principal Component Analysis (PCA) for dimensionality reduction
#step 3: transforming 
scaled_data = scaled.transform(df)
scaled_data


# In[297]:


#decomposition of data into 2 components.
#reason 2 components because breast cancer is either "malignant" or "benign"

pca = PCA(n_components  = 2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)


# In[298]:


#getting shape of new data (i.e. rows and columns)
scaled_data.shape
x_pca.shape


# In[299]:


#displaying array
x_pca


# In[300]:


#scatter plot for dimensionality reduction to 2 components
fig = plt.figure(figsize = (15,8))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x_pca[:,0],x_pca[:,1], c = "purple", s = 50)
ax.legend(['Malign'])

ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.view_init(30,120)


# In[ ]:




