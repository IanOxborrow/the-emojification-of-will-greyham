import cv2
import numpy as np


from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras import layers, callbacks, utils, applications, optimizers
from tensorflow.keras.models import Sequential, Model, load_model

physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass



def get_emoji(emoji_name, desired_size=(-1, -1)):
        
    emoji = cv2.imread('emojis/' + emoji_name + '.png')
    emoji_size = emoji.shape[0]
    emoji = cv2.cvtColor(emoji, cv2.COLOR_BGR2RGB)
    
    if desired_size[0] > -1:
        emoji = cv2.resize(emoji, (desired_size[0], desired_size[1]))

    emoji_feature_file = open('emojis/' + emoji_name + '.txt', "r")
    emoji_data = emoji_feature_file.read()
    emoji_feature_file.close()

    emoji_features = np.array(emoji_data.split(', '), dtype=np.float32).reshape(-1, 2)
    
    if desired_size[0] > -1:
        emoji_features = emoji_features * desired_size[0] / emoji_size

    return emoji, emoji_features


def get_face(face, desired_size=(-1, -1)):

    face_size = face.shape[0]
    
    processing_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    processing_face = cv2.resize(processing_face, (96, 96)) #reshape to 96,96 to be processed by the neural network
    processing_face = np.reshape(processing_face, (96, 96, 1)) / 255    
    processing_faces = np.array([processing_face])

    #print(processing_face.shape)
    #print(processing_face)
    
    print('about to load model')
    model = tf.keras.models.load_model('model_training/tensorflow_model/tf_model')
    print('loaded model')
    
    facial_features = model.predict(processing_faces)
    facial_features = facial_features.reshape(-1, 2)
    facial_features = facial_features * face_size / 96
    
    if desired_size[0] > -1:
        resized_face = cv2.resize(face, (desired_size[0], desired_size[1]))
        facial_features = facial_features * desired_size[0] / face_size

        
    resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

        
    return resized_face, facial_features