import argparse, os
import numpy as np

import tensorflow as tf
from keras import backend as K
import sys
# imports the model in model.py by name
from model import multiclassifier
sys.path.append('../../')
from helper.helper import data_preprocessing

## TODO: Complete the main code
if __name__ == '__main__':
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--parent_dir', type=str, default=os.environ['parent_direct'])

    parser.add_argument('--num_classes', type=str, default=os.environ['num_classes'])
    parser.add_argument('--image_height', type=str, default=os.environ['image_height'])
    parser.add_argument('--image_width', type=str, default=os.environ['image_width'])
    parser.add_argument('--batch_size', type=str, default=os.environ['batch_size'])

    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    parent_dir = args.parent_dir

    num_classes = args.num_classes
    img_height = args.image_height
    img_width = args.image_width 
    batch_size= args.batch_size

    # Data Preparation
    prep_obj = data_preprocessing()
    train_ds, val_ds = prep_obj.data_prep(parent_dir,training_dir, validation_dir, img_height, img_width,batch_size)

    # Instantiating the model
    print('\nModel Sub-Classing API')
    sub_class_model = multiclassifier(num_classes, img_height, img_width)
    sub_class_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = sub_class_model.fit(train_ds,validation_data=val_ds,epochs=epochs)

    path = '{}/my_classifier.h5'.format(model_dir)
    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    sub_class_model.save_weights(path)



