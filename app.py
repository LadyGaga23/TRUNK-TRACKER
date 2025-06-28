from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)
# model = load_model(r'F:\Amal_kalarickal\human-wildlife-conflict-master\human-wildlife-conflict-master\temperature_prediction_model_2.h5')
model = load_model('temperature_prediction_model_2.h5', compile=False)


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Failed to load image")
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB    image = img_to_array(image)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype("float32") / 255.0
    return image

def predict_temperature_and_presence(image_path):
    image = preprocess_image(image_path)
    predicted_temp = model.predict(image)[0][0]
     # Validate temperature range
    if not 34.0 <= predicted_temp <= 40.0:
        return predicted_temp, False, "Warning: Predicted temperature out of expected range (34.0–40.0°C)"
    elephant_present = predicted_temp >= 37.0
    return predicted_temp, elephant_present, None
    #elephant_present = predicted_temp >= 38.0
    #return predicted_temp, elephant_present

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        predicted_temp, elephant_present, warning = predict_temperature_and_presence(filepath)
        return render_template('result.html',
                              temperature=f"{predicted_temp:.2f}°C",
                              elephant="Elephant Detected ✅" if elephant_present else "No Elephant ❌",
                              warning=warning if warning else "")
    except ValueError as e:
        return f"Error processing image: {str(e)}", 400
    #predicted_temp, elephant_present = predict_temperature_and_presence(filepath)

    #return render_template('result.html',
    #                       temperature=f"{predicted_temp:.2f}°C",
     #                      elephant="Elephant Detected ✅" if elephant_present else "No Elephant ❌")



if __name__ == '__main__':
    app.run(debug=True)
