import tensorflow as tf
import tensorflow.python.keras
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_features(img_path,model):
    img=image.load_image(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img).flatten()
    nomalized_result=result/norm(result)


    return nomalized_result

filenames=[]

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

feature_list=[]

for file in filenames:
    feature_list.append(extract_features(file,model))

print(np.array(feature_list).shape)
