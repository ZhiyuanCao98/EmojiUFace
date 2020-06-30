import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from utils import *
from cnnModelBuild import *


# preprocess the data before we feed it into the convolutional neutral network model.
def preprocess_data():
    train_data, train_label = generate_train_sets()
    # transfer them to np array for easier tackling by the model
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    # Because the images are grayscale images, each pixel value have range from 0 to 255. 
    # Also, they have a dimension of 100 x 100. As a result, we should preprocess the 
    # data before you we use then in the model.
    train_data = train_data.reshape(-1, 100, 100, 1)

    # The data now is in an int8 format, 
    # before we use it in CNN, it should be converted to
    # float32 format and rescale the pixel values in range 0 - 1 inclusive
    train_data = train_data.astype('float32')
    train_data = train_data / 255.

    # We will convert the labels(categorical data) into a vector
    # of numbers (one hot encoding). Because machine algorithm 
    # only accept one boolean column for each category.
    train_label = to_categorical(train_label)

    # Partion the data correctly
    train_data, predict_data, train_label, predict_label = \
        train_test_split(train_data, train_label, test_size=0.1, random_state=13)

    return train_data, predict_data, train_label, predict_label


# We have three convolutional layers:
# (1) 32-3 x 3 filters 
# (2) 64-3 x 3 filters 
# (3) 128-3 x 3 filters 
# We also have three max-pooling layers, each of size is 2 * 2
# You'll use three convolutional layers:
def cnn_train_model():
    train_data, predict_data, train_label, predict_label = preprocess_data()

    fashion_model = initialize_cnn_model()

    # display different layers' training results
    # print(fashion_model.summary())

    # train the model
    fashion_train = fashion_model.fit(train_data, train_label, \
                                      batch_size=batch_size, epochs=epochs, verbose=1,
                                      validation_data=(predict_data, predict_label))

    # save it
    fashion_model.save("../models/fashion_model_dropout.h5py")


if __name__ == '__main__':
    cnn_train_model()
