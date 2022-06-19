import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers

class data_preprocessing():

    def perf_optimz(self):
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))

        return image_batch, labels_batch

    def data_prep(self, parent_dir, train_pth, val_pth, img_height, img_width, batch_size):
        try:
            train_ds = tf.data.experimental.load(train_pth)
            val_ds = tf.data.experimental.load(val_pth)

        except:
            train_ds = tf.keras.preprocessing.image_dataset_from_directory(parent_dir,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(img_height, img_width),
                batch_size=batch_size)
            
            val_ds = tf.keras.preprocessing.image_dataset_from_directory(paent_dir,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(img_height, img_width),
                batch_size=batch_size)

        return train_ds, val_ds

    
