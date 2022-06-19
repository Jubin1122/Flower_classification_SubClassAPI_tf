###  Deep Learning with Keras on Amazon SageMaker Local Mode

This tutorial shows how to classify images of flowers. It creates an image classifier using a tf.keras.Sequential model, and loads data using tf.keras.utils.image_dataset_from_directory.
Efficiently loading a dataset off disk.
Identifying overfitting and applying techniques to mitigate it, including data augmentation and dropout.
This tutorial follows a basic machine learning workflow:

Examine and understand data
* Build an input pipeline
* Build the model
* Train the model
* Test the model
Improve the model and repeat the process

#### Resources

* [Model Source](https://www.tensorflow.org/tutorials/images/classification)
* [Data Preparation](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory)
* [Git Lab](https://gitlab.com/juliensimon/aim410/-/blob/master/aim410.ipynb)

Typical Sturcture of the project

![alt text](pics\Strucutre_of_pro.png)

After downloading all the images, we will prepare the data object for the training.
Here, we are using a vanila style in sagemaker.
First of all we have to pass all the necessary environments variables, and *train.py* will be called. 

In *model.py* we will be writing only the infrastructure of the model. In this case scenario, I have used tensoflow sub-class api. You can also go with sequential or fuctional method.


*Predict.py* will capture the codes for classifying a flower, also will evaluate the validation accuracy.


**Why are these environment variables important anyway?**
Well, they will be automatically passed to our script by SageMaker, so that we know where the data sets are, where to save the model, and how many GPUs we have. So, if you write your local code this way, there won't be anything to change to run it on SageMaker.This feature is called 'script mode', it's the recommended way to work with built-in frameworks on SageMaker.

![alt text](pics\script_mode.png)

### Train on the notebook instance (aka 'local mode')

Train on the notebook instance (aka 'local mode')
Our code runs fine. Now, let's try to run it inside the built-in TensorFlow environment provided by SageMaker. For fast experimentation, let's use local mode.

![alt text](pics\local_mode.png)
