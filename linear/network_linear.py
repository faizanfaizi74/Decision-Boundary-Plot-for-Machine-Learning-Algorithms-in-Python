import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import svm

train_df = pd.read_csv('./data/train.csv')
np.random.shuffle(train_df.values)

print(train_df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df.x,train_df.y)

plt.scatter(train_df.x,train_df.y, marker='.', color=['red'])
plt.show()

# # Sequencial API (Seq: We can map One Input to One Output - Not Flexible)
# model = keras.Sequential([
# 	keras.layers.Dense(4, input_shape=(2,), activation='relu'), 
# 	keras.layers.Dense(2, activation='sigmoid')])

# Functional API (Func: It is more flexible as it can handle multiple input and multiple output - More Flexible)
inputs = keras.Input(shape=(2,))
x = keras.layers.Dense(4, activation='relu', name='First_layer')(inputs)
outputs = keras.layers.Dense(2, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())   # Network information -> Layers -> Nodes -> Parameters

model.compile(optimizer='adam', 
	          loss=keras.losses.SparseCategoricalCrossentropy(),
	          metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

model.fit(x, train_df.color.values, batch_size=4,  epochs=5, verbose=2)

test_df = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

print("EVALUATION")
model.evaluate(test_x, test_df.color.values)

plt.scatter(test_df.x,test_df.y, marker='*', color=['blue'])
plt.show()
print(test_df.x)
#------------------End Of File---------------------

#Plotting SVM Hyperplane and Margin