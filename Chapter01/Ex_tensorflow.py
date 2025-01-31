import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, losses, metrics


NB_CLASSES = 10
RESHAPED = 784
model = tf.keras.models.Sequential()
model.add(layers.Dense(NB_CLASSES, input_shape=(RESHAPED,), kernel_initializer="zeros", name='dense_layer', activation='softmax'))
