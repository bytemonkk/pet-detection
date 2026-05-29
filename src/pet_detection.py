import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

'''libraries'''
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

'''Preprocessing phase'''
'''Preprocessing the Training set'''
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    "D:/deep learning/ConvolutionalNeuralNetworks/dataset/training_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

'''Preprocessing the Test set'''
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    "D:/deep learning/ConvolutionalNeuralNetworks/dataset/test_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

'''Building The ConvolutionalNeuralNetworks'''
'''initializing The CNN'''
cnn = tf.keras.models.Sequential()

'''Convolution'''
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

'''Pooling'''
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

'''Adding a second Convolutional layer'''
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

'''Flattening'''
cnn.add(tf.keras.layers.Flatten())

'''Full Connection'''
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

'''Output layer'''
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

'''Training The ConvolutionalNeuralNetworks'''
'''compiling The CNN'''
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

'''Training The CNN and Evaluating it on the Test set'''
cnn.fit(x=training_set, validation_data=test_set, epochs=1)

'''Making a single prediction'''
import numpy as np
from keras_preprocessing import image
test_img = image.load_img("D:/deep learning/ConvolutionalNeuralNetworks/dataset/single_prediction/cat_or_dog_1.jpg", target_size=(64, 64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
res = cnn.predict(test_img)
training_set.class_indices
if res[0][0] == 1:
    prediction = 'dog'
else:
   prediction = 'cat'
print(prediction)