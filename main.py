# This project is using  Neural Networks and Functional API to train datasets from MNIST

# Implement the necessary TensorFlow and Keras modules to load and load the MINST dataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Loads the MNIST dataset that will be used for this Neural Network

# x_train and x_test contain the images
# y_train and y_test contain corresponding labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshapes the images into a flat vector and normalizes the pixel between 0 and 1
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# Sequential Keras API

# Enables the sequential Keras API that consists of three layer
# Two Layers are hidden: 512 and 216 neuron layers
# One output layer with 10 neurons
model = keras.Sequential(
    [
        # keras.Input(shape==(2))
        layers.Dense(512, activation='relu'),
        layers.Dense(216, activation='relu'),
        layers.Dense(10),

    ]
)

# Functional API

# Input layer has 784 neurons
# Two Hidden Layers: 512 and 256 neurons
# One output layer with 10 neurons
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(256, activation='relu')(inputs)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Configures the model for training: Specifies the loss function (Sparce Categorical Cross Entropy)
# as well as the evaluation rate which is the accuracy, and Adam which is
# the optimizer with 0.001 learning rate
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

# Trains the model using the training data with batch size of 32 and for 5 epochs
# Verbose is the progress bar level 2
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)

# Evaluates the trained model using the testing data and calculates the loss and accuracy
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
