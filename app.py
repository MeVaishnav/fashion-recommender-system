import tensorflow as tf
import tensorflow.python.keras
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_features(img_path,model):
