import platform
import glob
import argparse
import cv2

# global variables:

# all emotional pictures category folder link
emotion_dicts = glob.glob('../data/emotion/*')
# the current system: mac or windows
system = platform.system()
# all emotion categories
emotions = [e.split('\\')[-1] for e in emotion_dicts] if system == 'Windows' else [e.split('/')[-1] for e in
                                                                                   emotion_dicts]
# train data folder url
train_data_url = '../data/train_data'
# train data how many times
train_data_time = 5
# selected train model
train_model = "fisher"


def parse_arg():
    parser = argparse.ArgumentParser(description='Process to train the image classification model')
    parser.add_argument("-r", '--refresh', default=1.5, type=int, help="Camera emoji refresh time.", dest='r')
    parser.add_argument("-p", '--train_path', default=train_data_url,
                        help='path to the train_dataset')
    parser.add_argument("-n", default=train_data_time, type=int, help="train data n time(s).")
    # the only another choice of train model is convolutional neural network (cnn)
    parser.add_argument("-m", default=train_model, type=str, help="train model selected.")
    # parser.add_argument("-t", '--train_method', default='fisherface', 
    #     type=str, help="The algorithm used to train model (fisherface or cnn).")
    # parser.add_argument("-l", '--cnn_lib', default='pytorch', 
    #     type=str, help="which lib used to train CNN model.")   
      
    args = parser.parse_args()
    return args


# generate train data sets for training model
def generate_train_sets():
    train_data = []
    train_label = []
    for label_idx, emotion in enumerate(emotions):
        # find all images url under this category of emotion
        emotion_images = glob.glob(train_data_url + "/" + emotion + "/*")
        for image in emotion_images:
            gray_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
            train_data.append(gray_image)
        # create labels for the model, map to related images
        train_label.extend([label_idx] * len(emotion_images))

    return train_data, train_label
