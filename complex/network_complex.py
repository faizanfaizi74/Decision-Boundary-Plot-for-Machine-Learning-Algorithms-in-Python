import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('./data/train.csv')
np.random.shuffle(train_df.values)

print(train_df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df.x,train_df.y)

plt.scatter(train_df.x,train_df.y, marker='.', color=['red'])
plt.show()

# model = keras.Sequential([
# 	keras.layers.Dense(256, input_shape=(2,), activation='relu'),
# 	keras.layers.Dropout(0.4),
# 	keras.layers.Dense(256, activation='relu'),
# 	keras.layers.Dropout(0.4),
# 	keras.layers.Dense(256, activation='relu'),
# 	keras.layers.Dense(2, activation='sigmoid')])

# Functional API (Func: It is more flexible as it can handle multiple input and multiple output - More Flexible)
inputs = keras.Input(shape=(2,))
x = keras.layers.Dense(256, activation='relu', name='First_layer')(inputs)
#x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(256, activation='relu', name='Second_layer')(x)
#x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(256, activation='relu', name='Third_layer')(x)
outputs = keras.layers.Dense(2, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())   # Network information -> Layers -> Nodes -> Parameters

model.compile(optimizer='adam', 
	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	          metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

model.fit(x, train_df.color.values, batch_size=32, epochs=20)

test_df = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

print("EVALUATION")
model.evaluate(test_x, test_df.color.values)





