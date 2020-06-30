'''
Normalize the data | This file aims to preprocess the image dataset,
cut the face from the raw data and meet the requirements from the step of
trainning model
'''
import os
import glob
import cv2
import shutil
from detect_face import normalize_face
from os import path
from utils import *


def remove_old_train_dict():
    if path.exists(train_data_url):
        shutil.rmtree(train_data_url)
    os.mkdir(train_data_url)


def build_train_dict():
    for e in emotions:
        os.mkdir('%s/%s' % (train_data_url, e))


def generate_trainable_dataset():
    """
    This function can read raw image from ./data folder and 
    generate a train datasets which cut face out each dataset.
    """
    # remove the old train data folder
    remove_old_train_dict()
    # generate valid train data
    build_train_dict()

    print('==========\nGenerate train dataset\n==========')
    # loop categories of the emotions
    for i, emotion in enumerate(emotions):
        images = glob.glob(emotion_dicts[i] + '/*')
        count = 0

        # print names of all pictures and purify all pictures
        # that cannot be normalized
        path = emotion_dicts[i]
        files = [f for f in glob.glob(path + "**/*.*")]

        # loop each item in one specfici category
        for j, image in enumerate(images):
            image = cv2.imread(image)
            normalized_faces = normalize_face(image, True, files[j])
            # if one picture contains more than one face, we will loop them
            for idx, facezip in enumerate(normalized_faces):
                face = facezip[0]
                try:
                    cv2.imwrite('%s/%s/%s_%s.png' % (train_data_url, emotion, j, idx), face)
                    count += 1
                    # print('Picture: %s: %s_%s generated successfully!' % (emotion, j, idx))
                except:
                    print('Face no found in %s' % image)
        # check how many train data actually are created
        print('Category: %s , Total generated images: %s/%s' % (emotion, count, len(images)))
    print('==========\nAll dataset has been generated!\n==========')


if __name__ == '__main__':
    generate_trainable_dataset()
