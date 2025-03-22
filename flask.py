#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score


# In[2]:


filename =(r"C:\Users\Dakshith\Downloads\pima-indians-diabetes.data.csv")
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)


# In[4]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[8]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[9]:


gsv.best_params_ , gsv.best_score_


# In[10]:


clf = SVC(C= 15, gamma = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[11]:


import joblib
import pickle


# In[12]:


joblib.dump(clf,"model.pk!")


# In[ ]:


from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load('model.pk!')
@app.route('/')
def home():
    return 'welcome to the model API!'
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['featured']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})
if __name__ == '__main__':
    app.run(debug=True)

