# CS4100-final-project: EmojiUFace
Instantly recognize the facial expression and generate a corresponding emoji based on user emotion.

## Group Member:
Bowen Lei

Jiahao Cai

Zhiyuan Cao 

# Installation # 

### Requirements

  * Python 3.3+ 
  * macOS or Windows

### Installing module for the project

  * Install the foundemental module: run command ```pip3 install requirements.txt ```
  * If you have error: "Could not find a version that satisfies the requirement", please try the command: ```pip3 install -r requirements.txt```

# Usage #

### Run the project

  * Open the src folder: ```cd src```.
  * Run the project: ```python3 emojiUFace.py```,
    the default train model uses fisher face (Principal Components Analysis (PCA), a machine learning method).
  * optional train model: ```python3 emojiUFace.py -m "cnn"```, the cnn train model uses the convolutional neural network, which is a deep learning algorithm.
  
  * Hint!!! When you paly with our emojiUFace, please get closer to the camera for getting more accurate result.
### (OPTIONAL)

#### Upload and preprocess new images

  * Upload new human face images into correct categories, such as saving anger image in to the anger folder in the path ```./data/emotion/anger" ``` .
  * Preprocess all images and categorized them : ```python3 preprocessing_image.py```, which will preprocess all images and any image that cannot be recognized will be automatically removed.
  
#### Train new model

  * Train fisher face model: ```python3 train.py```, you also can set the times
  you want to train this model by running command: ```python3 train.py -n 10``` 
  (such as try to train this modal 10 times and pick up the highest accuracte model)
  * Train CNN (convolutional neural network) model: ```python3 cnnTrain.py```
  * Both trained model will be saved in the models folder

# Running Result #
One user:
![Screen Shot 2020-03-30 at 7 19 52 PM](https://media.github.ccs.neu.edu/user/2517/files/7ffd9f00-72bb-11ea-8c06-44da162cdfb4)
Multiple users: 
![Screen Shot 2020-04-13 at 1 50 40 PM](https://media.github.ccs.neu.edu/user/2517/files/0d49f480-7d8e-11ea-9dae-bfabf1a6c796)

# File Descriptions #
## detect_face.py:
We select the haar model from cv2 to implement the face detection function. 

The functions can identify the coordniates of all faces from the photo or stream. 

And it will automatically normalize the face images (color -> gray, resize)

## preprocess_image.py:

Before you run this file, you have to make sure the dataset in the 
```./data/emotion``` have more than one data folder. Each folder contains human face images that belong to the same category of the emotion. We call this categorization as the manually labelling images for later trainning model. Accepted format of the image can be JPG and PNG. 

Running the file will generate a series of normalized faces in the 
```./data/train_data``` path. All images will be saved into the related emotion folders that will be created based on recognized emotions. 

Before:
![image](https://media.github.ccs.neu.edu/user/2517/files/687be700-7d8e-11ea-8920-66a05695d0e4)
After:
![image](https://media.github.ccs.neu.edu/user/2517/files/8ea18700-7d8e-11ea-8e8d-e4fe34313161)


## train.py

Before you run this train file, it is necessary to have a well-design datasets and preprocess these images. In  other words, you have to make sure ```./data/train_data``` is not empty.

In this file, we use the fisherface classfication algoirthm to build the training model:
#### Fisherface 
http://www.scholarpedia.org/article/Fisherfaces
The algorithm attmptes to find a subspace which represents most of the data variance. This can be obtained with the use of Principal Components Analysis (PCA). When applied to face images, PCA yields a set of eigenfaces. These eigenfaces are the eigenvectors associated to the largest eigenvalues of the covariance matrix of the training data. The eigenvectors thus found correspond to the least-squares (LS) solution

The required data sets for this algorithm are train_images and train_labels. Training images have been preprocessed firstly and saved in train_data folder. Based on each image's label (folder name), we generated labels for each one. Classes of the label will present as an integer from 0 to N (number of emotions).

Then, we used sklearn module to create appropriate train sets and predicted sets. Train sets are used for training the fisher face model and predicted sets are used for evalutaing the accuracy of the model. 

Finally, we save the model the xml file and later we can use this model to predict the class (label) of given human face image.

# emojiUFace.py (Main File)

This file reloads saved models in the path ```./models``` and run the camera on the computer. It will capture user faces, used the model to predict the class of the currrent faces (find which emotion the users are presenting), and generate the related emoji to overlay user faces. 

## CNN TensorFlow Version:

### cnnTrain.py:  

The same preconditions of running the train.py. The required data sets for this algorithm are also train_images and train_labels. But, in this file, we used the convolutional neural network algorithm to build the model. When running this file, you can clearly observe the Q-learning process of training this model. The accuracy of actual data and predicted data will increase gradually by going through different layers again and again. 

### cnnModelBuild.py:

Initialize the convolutional neural network model by providing required configurations. We add different layers and max poolings for the neural network architecture. We even set the dropout to overcome overfitting with the regularization.

## CNN Pytroch Version:
### cnnPytorch.py

This file implement the CNN via Pytorch version. It requires user install ```pytorch``` first. It contains a class Net that has the Conv neural network and dropout procedures. This program allows user train a CNN model based on the train_data from our previous code.

## utils.py

It contains all global variables, such as the training data times and all classes of the emotion. The parse_arg function will make user can manually set the type of used training model and numbers of training fisher face model. The generate_train_sets function will generate training data sets and labels by loading data from train_data folder. 
