import numpy as np
import tensorflow as tf
from tensorflow import keras

# XOR input and output data
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 5000
model.fit(x_data, y_data, epochs=epochs, verbose=0)

# Test the trained XOR gate
predicted_output = model.predict(x_data)
predicted_output = (predicted_output > 0.5).astype(int)

for i in range(len(x_data)):
    print(f"Input: {x_data[i]}, Predicted Output: {predicted_output[i]}")
