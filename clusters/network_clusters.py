import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('./data/train.csv')
color_dict = {'red': 0, 'blue': 1, 'green': 2, 'teal': 3, 'orange': 4, 'purple': 5}
train_df['color'] = train_df.color.apply(lambda x: color_dict[x])
np.random.shuffle(train_df.values)

print(train_df.head())
print(train_df.color.unique())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df.x, train_df.y)

plt.scatter(train_df.x, train_df.y, marker='.', color=['red'])
plt.show()

# model = keras.Sequential([
# 	keras.layers.Dense(32, input_shape=(2,), activation='relu'),
# 	keras.layers.Dense(32, activation='relu'),
# 	keras.layers.Dense(6, activation='sigmoid')])

# Functional API (Func: It is more flexible as it can handle multiple input and multiple output - More Flexible)
inputs = keras.Input(shape=(2,))
x = keras.layers.Dense(32, activation='relu', name='First_layer')(inputs)
x = keras.layers.Dense(32, activation='relu', name='Second_layer')(x)
outputs = keras.layers.Dense(6, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())   # Network information -> Layers -> Nodes -> Parameters

model.compile(optimizer='adam', 
	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	          metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

model.fit(x, train_df.color.values, batch_size=4, epochs=10)

test_df = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

print("EVALUATION")
test_df['color'] = test_df.color.apply(lambda x: color_dict[x])
model.evaluate(test_x, test_df.color.values)

print("Prediction", np.round(model.predict(np.array([[-2,3]]))))





