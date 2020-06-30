import platform
import glob
import cv2
import time
import keras
import numpy as np
from detect_face import normalize_face
from utils import *
import time 

emotion_2_idx = dict()
idx_2_emotion = dict()
idx_2_emoji = dict()


def getFisherFaceModel():
    fisher_face = cv2.face.FisherFaceRecognizer_create()
    fisher_face.read('../models/emojiUFace.xml')

    return fisher_face


def getCNNmodel():
    cnn_model = keras.models.load_model("../models/fashion_model_dropout.h5py")

    return cnn_model


def get_predict_data(model, modelType, face):
    if (modelType == "fisher"):
        return model.predict(face)[0]
    elif (modelType == "cnn"):
        tackle_image = np.array([face])
        tackle_image = tackle_image.reshape(-1, 100, 100, 1)
        new_prediction = model.predict(tackle_image)
        predicted_class = np.argmax(np.round(new_prediction), axis=1)
        return predicted_class[0]


# run camera to recognize the user face
def run_cam(model, train_model_type, args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('No camera found!')
        exit()

    # get the related pictures with each label
    idx_2_emoji = dict()
    for k, item in idx_2_emotion.items():
        idx_2_emoji[k] = cv2.imread('../emoji/%s.png' % item, -1)

    interval_time = time.time()
    last_prediction = None
    # execute the detection of the user's face
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        for face, face_cor in normalize_face(frame):
            if time.time() - interval_time > args.r or last_prediction is None:
                prediction = get_predict_data(model, train_model_type, face)
                interval_time = time.time()
                last_prediction = prediction
                print(idx_2_emotion[prediction])
            x, y, w, h = face_cor
            onface_image = idx_2_emoji[prediction]
            # frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0))
            onface_image = cv2.resize(onface_image, (h, w))
            alpha_s = onface_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                frame[y:y + h, x:x + w, c] = (alpha_s * onface_image[:, :, c] + alpha_l * frame[y:y + h, x:x + w, c])
        
        # Display the resulting frame
        cv2.imshow('emojiUFace (Press q or 1 quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    for i, e in enumerate(emotions):
        emotion_2_idx[e] = i
        idx_2_emotion[i] = e

    args = parse_arg()
    train_model_type = args.m
    # default model is fisher
    model = getFisherFaceModel()

    if train_model_type == "cnn":
        model = getCNNmodel()

    run_cam(model, train_model_type, args)
