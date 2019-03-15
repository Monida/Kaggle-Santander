# -*- coding: utf-8 -*-
"""
Kaggle - Santander
"""
'''
This .py document has drafts of  models run to play and understand how the data 
reacts to the different models
'''

#Libraries
import numpy as np
import pandas as pd
import time


#Importing the dataset
train_set = pd.read_csv('../Data/train.csv')
test_set = pd.read_csv('../Data/test.csv')

x_train=train_set.iloc[:,2:202]
y_train=train_set.iloc[:,1]
x_test=test_set.iloc[:,1:201]

x_train.head()
y_train.head()
x_test.head()


'''
1st APPROACH
-Keras ANN
-Use all Data
-Without treating data imbalancing
'''

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_x_train = sc.fit_transform(x_train)
df_x_train=pd.DataFrame(scaled_x_train)
df_x_train.head()

scaled_x_test = sc.fit_transform(x_test)
df_x_test=pd.DataFrame(scaled_x_test)
df_x_test.head()

#Find best parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_model(optimizer):
    model = Sequential()
    model.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu', input_dim=200))
    model.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn = build_model)
parameters = {'batch_size': [25,50],
             'epochs': [10, 20],
             'optimizer':['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = model,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10)
start_time = time.time()
grid_search = grid_search.fit(scaled_x_train,y_train)
elapsed_time = (start_time - time.time())/60

#Best parameters
grid_search.best_params_
''' {'batch_size': 25, 'epochs': 10, 'optimizer': 'rmsprop'} '''

grid_search.best_score_
'''0.90634'''

#Fit best model
best_model.fit(scaled_x_train,y_train, batch_size =25, epochs = 10)

#Predic with best model
predictions = best_model.predict_proba(scaled_x_test)

predictions=predictions>0.5

#Save the model
import os
script_dir = os.path.dirname('Kaggle - Santander')
model_backup_path = os.path.join('kaggle_santander_model.h5')
best_model.save(model_backup_path)

#Load the model
from keras.models import load_model
best_model = load_model('kaggle_santander_model.h5')

#Predict
predictions = best_model.predict_proba(scaled_x_test)

threshold = 0.5
predictions = np.where(predictions<threshold,0,1)
predictions

np.count_nonzero(predictions==1)

#Write predictions in the right format

import submission_format
df = submission_format(predictions)
df.head()

#Save as csv
df.to_csv('santander_predictions_2nd_approach.csv', sep=',',index=False)

# Making the Confusion Matrix to evaluate the model 
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_train)
#Cannot use confusion matrix because the test set doesn't have y_test
'''
Conclusions:
Score: 0.0597
1. Imabalnced data. 0 to 1 ratio is 180K:2K --> 90:1

'''

'''
2nd APPROACH
-Lower the threshold
-Without treating data imbalancing
'''
#Lower the threshold
threshold = 0.2
predictions = np.where(predictions<threshold,0,1)
predictions

np.count_nonzero(predictions==1)

'''
Conclusion
Score: 0.673
-It  increased the number of 1's but not accurately since the threshold let more 1's be
accepted, but the training was still done on imbalanced data
'''

'''
3rd APPORACH
-If the 0 to 1 ratio in the training set is 10:1, remove enough 0 data to make ratio 5:1
-Remove 90K 0's
-Train data with new undersampled dataset
'''
#Make the dataset balanced

extra0s=y_train[y_train==0].sample(90000)
y_train_3=y_train[~y_train.index.isin(extra0s.index)]
scaled_x_train_3 = pd.DataFrame(scaled_x_train)

scaled_x_train_3 = scaled_x_train_3.drop(scaled_x_train_3.index[extra0s.index])
    

#train the model
model_3 = Sequential()
model_3.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu', input_dim=200))
model_3.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu'))  
model_3.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model_3.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fit best model
model_3.fit(scaled_x_train_3,y_train_3, batch_size =25, epochs = 10)
    
#predict values
predictions = model_3.predict_proba(scaled_x_test)

# Evaluate the model

#Save the model
import os
script_dir = os.path.dirname('Kaggle - Santander')
model_backup_path = os.path.join('kaggle_santander_model_3.h5')
model_3.save(model_backup_path)

#Write predictions in the right format
import submission_format

df = submission_format(predictions)
df.head()

#Saves as csv
df.to_csv('santander_predictions_3rd_approach.csv', sep=',',index=False)

'''
Conclusions
Score:0.659
-Did not improve much more than lowering the threshold
'''

'''
4th APPROACH
-Model of 1st approach
-Treating imbalanced data with boot strapping
-Dividing train_set into train_set & eval_set(evaluation)
'''
from sklearn.utils import resample

train_set = pd.read_csv('train.csv')
train_set = train_set.iloc[:,1:202]
test_set = pd.read_csv('test.csv')


newtrain_set = resample(train_set, replace=True, n_samples = 160000, random_state=1)
neweval_set = resample(train_set, replace=True, n_samples = 40000, random_state=1)

x_train_4 = newtrain_set.iloc[:,1:201]
y_train_4 = newtrain_set.iloc[:,0]

x_eval_4 = neweval_set.iloc[:,1:201]
y_eval_4 = neweval_set.iloc[:,0]

x_test_4 = test_set.iloc[:,1:201]

#Feature scaling
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
scaled_x_train_4 = sc.fit_transform(x_train_4)
scaled_x_eval_4 = sc.fit_transform(x_eval_4)
scaled_x_test_4 = sc.fit_transform(x_test_4)
    
#train the model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


model_4 = Sequential()
model_4.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu', input_dim=200))
model_4.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu'))  
model_4.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model_4.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fit best model
model_4.fit(scaled_x_train_4,y_train_4, batch_size =25, epochs = 10)
    
#predict eval_values

import submission_format

predictions = model_4.predict_proba(scaled_x_eval_4)
[predictions,number_of_ones] = submission_format.right_format(predictions,40000,0.2)
y_eval_4 = y_eval_4.values
predictions = predictions.values

# Evaluate the model AUC ROC
TP=0
FP=0
TN=0
FN=0

for i in range(40000):
    if predictions[i][1]==y_eval_4[i]==0:
        TN+=1
    if predictions[i][1]==y_eval_4[i]==1:
        TP+=1
    if predictions[i][1]!=y_eval_4[i] and predictions[i][1]==0:
        FN+=1
    if predictions[i][1]!=y_eval_4[i] and predictions[i][1]==1:
        FP+=1
        
Sensitivity = TP/(TP + FN)
Specificity = TN/(TN + FP)

roc_value=roc_curve(y_eval_4,predictions)
auc=roc_auc_score(y_eval_4,predictions)

#predict test_values
predictions = model_4.predict_proba(scaled_x_test_4)

#Write predictions in the right format

[predictions,number_of_ones] = submission_format.right_format(predictions,200000,0.2)

#Save the model
import os
script_dir = os.path.dirname('Kaggle - Santander')
model_backup_path = os.path.join('kaggle_santander_model_4.h5')
model_4.save(model_backup_path)

#Saves as csv
predictions.to_csv('santander_predictions_4th_approach.csv', sep=',',index=False)

'''
Conclusions
Score:0.64
-auc
-Did not improve much more than 2nd approach 
'''

'''
5th APPROACH
-Model of 1st approach
-Treating imbalanced data with cross-validation
-Dividing train_set into train_set & eval_set(evaluation) using train_test_split
-Train the model 20 times using boostrapping sample of 1/4 of the train_set
-Change ANN to 
input layer 200 input
2nd layer 400
3rd layer 200
4th layer 100
5th layer 50
last layer 1
'''
# Splitt the dataset into the raining set and evaluation set
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

x_train_5, x_eval_5, y_train_5, y_eval_5 = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

#Take all the 1's of the y_train_5 and match them the same amount of 0's
y_train_5_1s = y_train_5[y_train_5==1]
y_train_5_0s = y_train_5[y_train_5!=1]

x_train_5_1s=x_train_5.loc[y_train_5_1s.index,:]
x_train_5_0s=x_train_5.loc[y_train_5_0s.index,:]

#Feature scaling
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
#Teach the SC to adjust the data based on the training set
sc.fit(x_train_5)
scaled_x_eval_5 = sc.transform(x_eval_5)
scaled_x_test_5 = sc.transform(x_test)
    
#train the model
'''
If sensitivity > 90

Randomly take 1/4/a and train model
Do this 20 times

Find predictions with test set
'''
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

model_5 = Sequential()
model_5.add(Dense(units=200, kernel_initializer = 'uniform', activation = 'relu', input_dim=200))
model_5.add(Dense(units=400, kernel_initializer = 'uniform', activation = 'relu'))
model_5.add(Dense(units=200, kernel_initializer = 'uniform', activation = 'relu'))
model_5.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu'))
model_5.add(Dense(units=50, kernel_initializer = 'uniform', activation = 'relu'))
model_5.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model_5.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fit best model 20 times on samples of 1/4 of the scaled_x_train_5 data

for i in range(20):
    y_1s = resample(y_train_5_1s,replace=False,n_samples = 1000)
    x_1s = x_train_5_1s.loc[y_1s.index]
    y_0s = resample(y_train_5_0s,replace=False,n_samples = 1000)
    x_0s = x_train_5_0s.loc[y_0s.index]
    
    y=pd.concat([y_0s,y_1s])
    y=y.sort_index()
    
    x=pd.concat([x_0s,x_1s])
    x=x.sort_index()
    
    scaled_x = sc.transform(x)
    model_5.fit(scaled_x,y, batch_size = 20, epochs = 10)

#predict eval_values
predictions = model_5.predict_proba(scaled_x_eval_5)
threshold = 0.2
predictions = np.where(predictions<threshold,0,1)

y_eval_5 = y_eval_5.values

# Evaluate the model AUC ROC
TP=0
FP=0
TN=0
FN=0

for i in range(40000):
    if predictions[i][0]==y_eval_5[i]==0:
        TN+=1
    if predictions[i][0]==y_eval_5[i]==1:
        TP+=1
    if predictions[i][0]!=y_eval_5[i] and predictions[i][0]==0:
        FN+=1
    if predictions[i][0]!=y_eval_5[i] and predictions[i][0]==1:
        FP+=1
        
Sensitivity = TP/(TP + FN)
Specificity = TN/(TN + FP)

roc_value=roc_curve(y_eval_5,predictions)
auc=roc_auc_score(y_eval_5,predictions)

'''
If sensitivity > 90

Randomly take 1/4/a and train model
Do this 20 times

Find predictions with test set
'''

#predict test_values
predictions = model_5.predict_proba(scaled_x_test_5)

#Write predictions in the right format
import submission_format
[predictions,number_of_ones] = submission_format.right_format(predictions,200000,0.2)

#Save the model
import os
script_dir = os.path.dirname('Kaggle - Santander')
model_backup_path = os.path.join('kaggle_santander_model_5.h5')
model_5.save(model_backup_path)

#Saves as csv
predictions.to_csv('santander_predictions_5th_approach.csv', sep=',',index=False)

'''
Conclusions:
Score: 0.731
'''

'''
6th APPROACH
-Logisti regression
-With data treatment as in 5th APPROACH

'''
# Splitt the dataset into the raining set and evaluation set
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

x_train_6, x_eval_6, y_train_6, y_eval_6 = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

#Take all the 1's of the y_train_6 and match them the same amount of 0's
y_train_6_1s = y_train_6[y_train_6==1]
y_train_6_0s = y_train_6[y_train_6!=1]

x_train_6_1s=x_train_6.loc[y_train_6_1s.index,:]
x_train_6_0s=x_train_6.loc[y_train_6_0s.index,:]

#Feature scaling
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
#Teach the SC to adjust the data based on the training set
sc.fit(x_train_6)
scaled_x_eval_6 = sc.transform(x_eval_6)
scaled_x_test_6 = sc.transform(x_test)
    
#Implement logistic Regression
from sklearn.linear_model import LogisticRegression

model_6 = LogisticRegression()

for i in range (50):
    y_1s = resample(y_train_6_1s,replace=False,n_samples = 1000)
    x_1s = x_train_6_1s.loc[y_1s.index]
    y_0s = resample(y_train_6_0s,replace=False,n_samples = 1000)
    x_0s = x_train_6_0s.loc[y_0s.index]
    
    y=pd.concat([y_0s,y_1s])
    y=y.sort_index()
    
    x=pd.concat([x_0s,x_1s])
    x=x.sort_index()
    
    scaled_x = sc.transform(x)
    model_6.fit(scaled_x,y)

#Prediction
predictions=model_6.predict_proba(x_eval_6)
threshold = 0.7
predictions = np.where(predictions<threshold,0,1)

y_eval_6 = y_eval_6.values

# Evaluate the model AUC ROC
TP=0
FP=0
TN=0
FN=0

for i in range(40000):
    if predictions[i][0]==y_eval_6[i]==0:
        TN+=1
    if predictions[i][0]==y_eval_6[i]==1:
        TP+=1
    if predictions[i][0]!=y_eval_6[i] and predictions[i][0]==0:
        FN+=1
    if predictions[i][0]!=y_eval_6[i] and predictions[i][0]==1:
        FP+=1
        
Sensitivity = TP/(TP + FN)
Specificity = TN/(TN + FP)

roc_value=roc_curve(y_eval_6,predictions)
auc=roc_auc_score(y_eval_6,predictions)

'''
7th APPROACH
'''
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
scaled_x_train = sc.transform(x_train)
df_x_train=pd.DataFrame(scaled_x_train)

scaled_x_test = sc.transform(x_test)
df_x_test=pd.DataFrame(scaled_x_test)

#Find best parameters
from keras.models import Sequential
from keras.layers import Dense

model_7 = Sequential()
model_7.add(Dense(units=200, kernel_initializer = 'uniform', activation = 'relu', input_dim=200))
model_7.add(Dense(units=400, kernel_initializer = 'uniform', activation = 'relu'))
model_7.add(Dense(units=200, kernel_initializer = 'uniform', activation = 'relu'))
model_7.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu'))
model_7.add(Dense(units=50, kernel_initializer = 'uniform', activation = 'relu'))
model_7.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model_7.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fit model
model_7.fit(scaled_x_train,y_train, batch_size =25, epochs = 10)

#Predic with best model
predictions = model_7.predict_proba(scaled_x_test)
threshold = 0.5
predictions = np.where(predictions<threshold,0,1)
predictions

np.count_nonzero(predictions==1)

#Write predictions in the right format

import submission_format
[predictions,number_of_ones] = submission_format.right_format(predictions,200000,0.2)

#Save as csv
predictions.to_csv('santander_predictions_7th_approach.csv', sep=',',index=False)

#Save the model
import os
script_dir = os.path.dirname('Kaggle - Santander')
model_backup_path = os.path.join('kaggle_santander_model_7.h5')
model_7.save(model_backup_path)


'''
Conclusions
Score: 0.651
'''

'''
8th APPROACH
-XGBoost
'''
import xgboost as xgb
import pandas as pd
from sklearn.utils import resample
import numpy as np

#Import the dataset
train_set = pd.read_csv('../Data/train.csv')
test_set = pd.read_csv('../Data/test.csv')

newtrain_set = resample(train_set, replace=True, n_samples = 160000, random_state=1)
neweval_set = resample(train_set, replace=True, n_samples = 40000, random_state=1)

train_8 = newtrain_set.iloc[:,2:202]
label = newtrain_set.iloc[:,1]
train_8 = xgb.DMatrix(train_8, label=label)

eval_8 = neweval_set.iloc[:,2:202]
label = neweval_set.iloc[:,1]
eval_8 =xgb.DMatrix(eval_8, label=label)

test_8 = test_set.iloc[:,1:201]
test_8 = xgb.DMatrix(test_8)

#Set parameters
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric':'auc'}
evallist = [(eval_8, 'eval'), (train_8, 'train')]

#Fit model
num_round = 200
bst = xgb.train(param, train_8, num_round, evallist)
bst.save_model('../Models/kaggle_santander_model_8.model')

#Predictions
predictions = bst.predict(test_8)

threshold = 0.2
predictions = np.where(predictions<threshold,0,1)
predictions

np.count_nonzero(predictions==1)

#Write predictions in the right format

import submission_format
[predictions,number_of_ones] = submission_format.right_format(predictions,200000,0.5)

#Save as csv
predictions.to_csv('../Data/santander_predictions_8th_approach.csv', sep=',',index=False)


'''
CONCLUSION
-Score: 0.746
The score improved but when evaluating the model it found 26388 1's out of 200000 
(0:1 ratio is about 90:1), similar ration than in the original training set.
However, since the score was still low, it means that the observations classified as 1 as 
wrong and the model might be overfitted
'''

'''
9th APPROACH
-XGBoost
-Data undersampling
'''
import xgboost as xgb
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import numpy as np

#Import the dataset
train_set = pd.read_csv('../Data/train.csv')
test_set = pd.read_csv('../Data/test.csv')

x_train= train_set.iloc[:,2:202]
y_train = train_set.iloc[:,1]
test_9 = test_set.iloc[:,1:201]
test_9 =  xgb.DMatrix(test_9)

# Splitt the dataset into the raining set and evaluation set
x_train_9, x_eval_9, y_train_9, y_eval_9 = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

y_train_9=pd.DataFrame(y_train_9)
train_9 = y_train_9.join(x_train_9)

y_eval_9 = pd.DataFrame(y_eval_9)
eval_9 = y_eval_9.join(x_eval_9)
label = eval_9.iloc[:,0]
eval_9 = xgb.DMatrix(eval_9,label=label)

#Take all the 1's of the y_train_9 and match them the same amount of 0's
train_9_1s = train_9[train_9['target']==1]
train_9_0s = train_9[train_9['target']!=1]

#Set parameters
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric':'auc'}

#Fit model 20 times with 1/4 of the of the scaled_x_train_9 data
num_round = 100

list_evallist = []

for i in range (20):
    train_1s = resample(train_9_1s, replace = False, n_samples = 4000)
    train_0s = resample(train_9_0s, replace = False, n_samples = 4000)
    
    train_9 = pd.concat([train_1s,train_0s])
    train_9 = train_9.sort_index()
    label = train_9.iloc[:,0]
    train_9 = xgb.DMatrix(train_9,label)
    
    evallist = [(eval_9, 'eval'), (train_9, 'train')] 
    list_evallist.append(evallist)
    
    bst = xgb.train(param, train_9, num_round, evallist)
    
    
bst.save_model('../Models/kaggle_santander_model_9.model')


#Predictions
predictions = bst.predict(eval_9)

threshold = 0.2
predictions = np.where(predictions<threshold,0,1)
predictions

np.count_nonzero(predictions==1)

predictions = bst.predict(test_9)

threshold = 0.2
predictions = np.where(predictions<threshold,0,1)
predictions

np.count_nonzero(predictions==1)

#Write predictions in the right format

import submission_format
[predictions,number_of_ones] = submission_format.right_format(predictions,200000,0.5)

#Save as csv
predictions.to_csv('../Data/santander_predictions_8th_approach.csv', sep=',',index=False)



