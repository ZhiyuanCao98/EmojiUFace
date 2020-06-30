"""
This file contains the method to detect the face in a picture
"""
import cv2
import os

faceCascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')


def faces_cordinate(img, scaleFactor=1.3, minNeighbors=5):
    faces_cor = faceCascade.detectMultiScale(
        img,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(50, 50)
    )

    return faces_cor


def normalize_face(img, do_purify=False, img_url=""):
    faces_cor = faces_cordinate(img)
    normalized_face_imgs = []
    for (x, y, w, h) in faces_cor:
        face = (img[y:y + h, x:x + w])
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resize_face = cv2.resize(gray_face, (100, 100))
        normalized_face_imgs.append(resize_face)

    # find all images that cannot be normalized, and delete them
    if do_purify and len(faces_cor) == 0:
        print(img_url)
        os.remove(img_url)

    return zip(normalized_face_imgs, faces_cor)
