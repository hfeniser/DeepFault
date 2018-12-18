from keras.models import Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Activation
from utils import load_MNIST
from sklearn.model_selection import train_test_split
import numpy

seed = 7

# Construct model
model = Sequential()

# Add input layer.
# MNIST dataset: each image is a 28x28 pixel square (784 pixels total).
model.add(Flatten(input_shape=(1, 28, 28)))

# Add hidden layers.
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))

# Add output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print information about the model
print(model.summary())

X_train, Y_train, X_test, Y_test = load_MNIST()
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                  test_size=1/6.0,
                                                  random_state=seed)

# Fit the model
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

# Save the model
model_json = model.to_json()
with open('mnist_test_model.json', "w") as json_file:
    json_file.write(model_json)

model.save_weights("mnist_test_model.h5")
