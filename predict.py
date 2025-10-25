import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

class_names = [
    'leaves_powdery_mildew',     
     'pepper__bell__bacterial_spot',
     'pepper__bell__healthy',
     'Potato___Early_blight',
     'Potato___healthy',
     'Potato___Late_blight',
     'Tomato__Target_Spot ',
     'Tomato__Tomato_blight',
     'Tomato__Tomato_mosaic_virus',
     'Tomato__Tomato_YellowLeaf__Curl_Virus',
     'Tomato_Bacterial_spot',
     'Tomato_Septoria_leaf_spot',
     'Tomato_Spider_mites_Two_spotted_spider_mite'
     'tomato_tomato_early rust',
     'tomato_tomato_healthy'

    
    
]

# Load and preprocess the image
img = cv2.imread('test_leaf.jpg')  # your test image
img = cv2.resize(img, (128, 128))  # must match training input size
img = img / 255.0                  # normalize
img = np.expand_dims(img, axis=0) # add batch dimension


prediction = model.predict(img)
predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]

print(" Predicted Class:", predicted_class)
