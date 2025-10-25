from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model("plant_disease_model.h5")

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classes = ['leaves_powdery_mildew',     
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

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/home', methods=['POST'])
def home():
    username = request.form['username']
    password = request.form['password']
    if username == 'admin' and password == 'admin':
        return render_template('predict.html')
    else:
        return render_template('login.html', error="Invalid credentials")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    result = model.predict(img_array)
    predicted_class = classes[np.argmax(result)]

    return render_template('predict.html', prediction=predicted_class, image_file=filepath)

if __name__ == '__main__':
    app.run(debug=True)
