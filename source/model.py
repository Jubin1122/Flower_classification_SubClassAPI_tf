import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

class multiclassifier(tf.keras.Model):
    def __init__(self, num_classes, img_height, img_width):
        super(multiclassifier, self).__init__()
        # define all layers in init
        
        # data augment
        self.rndmflip = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(img_height,img_width,3))
        self.rndmrotate = tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)
        self.rndmzoom = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3))
        # Layer of block 1
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu")
        self.max1 = tf.keras.layers.MaxPooling2D()
        
        # Layer of block 2
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu")
        self.max2 = tf.keras.layers.MaxPooling2D()

        # Layer for block3
        self.conv3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu")
        self.max3 = tf.keras.layers.MaxPooling2D()
    
        # Flattening, followed by classifier
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, input_tensor, training=False):
        # Data Augmentation
        x = self.rndmflip(input_tensor)
        x = self.rndmflip(x)
        x = self.rndmzoom(x)
        x= self.rescale(x)
        # forward pass: block 1 
        x = self.conv1(x)
        x = self.max1(x)

        # forward pass: block 2 
        x = self.conv2(x)
        x = self.max2(x)

        # forward pass: block 3 
        x = self.conv3(x)
        x = self.max3(x)

        # Flattening, followed by classifier
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense(x)