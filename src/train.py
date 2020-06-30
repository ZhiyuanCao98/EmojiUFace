import glob
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from utils import *


# Principal Components Analysis (PCA) for computer vision
# An important machine learning method for dimensionality reduction is called Principal Component Analysis.
def train_model(fishface):

    # create trainset, labels,prediction trainset , and prediction labels for training model
    train_set, prediction_data, train_labels, prediction_labels = \
        train_test_split(train_images, labels, test_size=0.1, shuffle=True)

    #print("train_set: %s, prediction_data: %s, train_lable: %s, predicted_labels: %s" % (train_set.shape, prediction_data.shape, train_labels.shape, prediction_labels.shape))

    # Execute the model training
    fishface.train(train_set, np.asarray(train_labels))
    accuracy = sum([1 for i, image in enumerate(prediction_data) \
                    if fishface.predict(image)[0] == prediction_labels[i]]) / len(prediction_labels)
    print("The accuracy of the model is: ", accuracy)

    return accuracy


if __name__ == '__main__':
    args = parse_arg()
    train_data_time = args.n
    currentAccuracy = float("-inf")

     # get the train sets:
    train_images, labels = generate_train_sets()

    # initialize the fisher face recognizer
    fishface = cv2.face.FisherFaceRecognizer_create()
    for i in range(train_data_time):
        print("Training attempt ", i+1, ":")
        accuracy = train_model(fishface)
        if currentAccuracy < accuracy:
            currentAccuracy = accuracy
            fishface.save('../models/emojiUFace.xml')
            print('======== Successfully saved!')

    print('Finish the model training! Current accuracy is %s ========' % currentAccuracy)
