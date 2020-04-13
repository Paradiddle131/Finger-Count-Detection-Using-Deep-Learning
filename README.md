# Translating Sign Language to Text and Audio 

Some headers are explanied briefly below:

- [Core Functions](#core-functions)
- [Download Data](#download-data)
- [More Data](#more-data)
- [Extract Frames From Videos](#extract-frames-from-videos)
- [Data Analysis](#data-analysis)
- [Dataset Preparation](#dataset-preparation)
- [Data Augmentation](#data-augmentation)
- [Model](#model)
- [Evaluation](#evaluation)

## Core Functions

Some of the file explorer management methods of python modules such as `os` and `shutil` have been modified according to the needs.

## Download Data

Video data is downloaded from google drive of the author either by mounting drive or using public drive link. [This Google Drive Link](https://drive.google.com/open?id=143LEc5sai_ReSzNSxKrkXKxH5iZl_XL4) is where the video data is stored.

![image](https://user-images.githubusercontent.com/36932448/79135379-ea343a80-7db7-11ea-9c90-8d07c99c3a26.png)

## More Data

A [fingers dataset](https://www.kaggle.com/koryakinp/fingers) published in Kaggle may be used later by concatinating with the custom dataset.

![image](https://user-images.githubusercontent.com/36932448/79135508-118b0780-7db8-11ea-8dd6-1363229a7c3f.png)

## Extract Frames From Videos

Many videos of different people showing their hands in front of a camera doing signs of numbers from 1 to 5 is recorded. Now they are going to be split into frames and saved as images with following format: 

`"{finger-no}_{frame-no}.jpg"`

## Data Analysis

Extracted frames are analysed in this section. Their dimensions and ratios are found and processed, datasets are merged if there are more than one.

![image](https://user-images.githubusercontent.com/36932448/79135554-32ebf380-7db8-11ea-9c60-ca387499f223.png)

## Dataset Preparation

In this section, noisy raw data is being processed by converting into grayscale, applying Gaussian Blur _(for background noise removal)_ and used Adaptive Threshold to detect the hand gesture and its contours. They converted into numpy arrays and finally split into train and test sets with 20% test ratio before feeding into the CNN model by using `train_test_split` method of [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). 

Labels are also extracted from the corresponding file name and saved as a numpy array using `prepare_dataset(folder)` method.

For the sake of regularization of dataset, all images containing integers with uint8 data type (0-255) normalized into 0-1 scale by dividing by 255.0.

Since the labels are categorical, `(either 1, 2, 3, 4 or 5)` the One Hot Encoding technique is used to convert the labels into `[1, 0, 0, 0, 0]` format using `to_categorical` method of [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical).

## Data Augmentation

Since we have limited amount of data relative to the need of a Convolutional Neural Network that can predict the signing accurately, we used "Data Augmentation" technique to achieve diversity in terms of color, shape, rotation, zoom etc. by using `ImageDataGenerator` of [Keras](https://keras.io/preprocessing/image/)

## Model

In the first cell after the library imports, necessary configuration is done such as defining batch size, number of epochs that we are going to train the model, learning rate and number of classes that the CNN model is expected to predict.

Then, we define the `create_model()` function with the CNN architecture. This will return the model itself when we call it.
After checking its architecture and parameters in every layer by using `model.summary()` method, we need to compile the model with an optimizer and loss function.

Right before the training starts, some helpful [Callback Tools of Keras](https://keras.io/callbacks/#modelcheckpoint) are going to assist the training process.

- `ModelCheckpoint`: Checkpoints are going to be saved after every epoch during the training to be able to continue where the model left trraining if something went wrong.
- `ReduceLROnPlateau`: When the loss could not be reduced on the last epoch, the learning rate will be reduced at some factor to prevent underfitting.
- `EarlyStopping`: When the validation loss could not be reduced on the last epoch, the training process is going to be stopped to prevent wasting resource.

### Evaluation

After the training completed, metrics of the model will be monitored and evaluated using [Matplotlib plot methods](https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.html) and tested with new data.

Results are expected to be like the following evaluation:

![fingers](https://user-images.githubusercontent.com/36932448/79144172-30dd6100-7dc7-11ea-94e0-120582bc4112.gif)
