# Kaggle---Santander
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

