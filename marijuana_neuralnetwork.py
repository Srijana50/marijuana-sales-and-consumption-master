# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/srija/OneDrive/Desktop/Lab 4 - ANN/marijuana_dataset.csv')
#numerical_variable = ['income','distance','quantity','quality_score','legal','dayspostlegal']
numerical_variable = ['distance','quantity','quality_score','legal','dayspostlegal']
X=dataset.loc[:,numerical_variable]
y = dataset.iloc[:, 10].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
print(y_train.mean())
print(y_test.mean())

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building MLP for predicting ppg

## Create standard ANN sequential model - sequential layers of neurons

#Part 2- lets make the ANN

# importing keras libraries and packages
import keras
from keras.models import Sequential
## We will use fully connected (dense) layers.  All neurons are fully connected with previous layer
from keras.layers import Dense

#initialising Ann classifier by defining it as sequence of layer
nn_reg = Sequential()
## Input (first) layer of neurons is created for you.Then add the first hidden layer
## n_input = number of independent variable
n_input = X_train.shape[1]

n_hidden = 32
# first hidden layer
nn_reg.add(Dense(units=6, activation='relu', input_shape=(n_input,)))
# add second hidden layer
nn_reg.add(Dense(units=12, activation='relu'))

# add third hidden layer
nn_reg.add(Dense(units=n_hidden, activation='relu'))
# output layer - only one layer because we only want one value (price prediction)
nn_reg.add(Dense(units=1, activation=None))


#Training the neural network
## compiling step
## Objective is to gradually reduce mean squared error in each iteration.
## ADAM optimizer adjusts learning rates to find best learning rates for each parameter
nn_reg.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae','accuracy'])
nn_reg.summary()


batch_size = 64
n_epochs = 250
history =nn_reg.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size , validation_split=0.2)
print(history.history.keys())

test_predicted = nn_reg.predict(X_test)
residual = test_predicted-y_test



from sklearn.metrics import mean_squared_error
y_pred_train = nn_reg.predict(X_train)
y_pred_test = nn_reg.predict(X_test)
train_mse = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
test_mse = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
print("Train MSE: {:0.3f} \nTest MSE: {:0.3f}".format(train_mse, test_mse))



fig, ax = plt.subplots(figsize=(8,5))
ax.plot(np.log(history.history['loss']), label='Training Loss')
ax.plot(np.log(history.history['val_loss']), label='Validation Loss')
ax.set_title("log(Loss) vs. epochs", fontsize=15)
ax.set_xlabel("epoch number", fontsize=14)
ax.legend(fontsize=12)
ax.grid();

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(history.history['mean_absolute_error'], label='Train MAE')
ax.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
ax.set_title("MAE vs. epochs", fontsize=15)
ax.set_xlabel("epoch number", fontsize=14)
ax.legend(fontsize=12)
ax.grid();

n_hidden =32
# defining a function build_regressor. build_regressor creates and returns the Keras sequential mode
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=n_hidden,activation='relu', input_dim=n_input))
    regressor.add(Dense(units=n_hidden, activation='relu'))
    regressor.add(Dense(units=1, activation=None))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])
    return regressor


#passing build_regressor function to the build_fn argument when constructing the KerasRegressor class. Batch_size is 32 and we run 100 epochs
from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=build_regressor, batch_size=64 ,epochs=100)

# fitting model in training data
results=regressor.fit(X_train,y_train)

## predicting the ppg
y_pred= regressor.predict(X_test)

residual =  y_test-y_pred


import seaborn as sns
sns.distplot(residual)

from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residual, X_test)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))


