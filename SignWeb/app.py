from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

app = Flask(__name__)
model = load_model('sign_language_model.h5')
labels = os.listdir(r'C:\Users\Bhuvana R\Documents\datasets_zip\datasets')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        encoded_data = data.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Debug: Print image shape
        print("Input Image Shape:", img.shape)
        
        # Make sure the image is preprocessed similarly to how it was during training
        
        prediction = model.predict(np.array([img]))
        
        # Debug: Print prediction probabilities
        print("Prediction Probabilities:", prediction)
        
        gesture = labels[np.argmax(prediction)]
        
        return jsonify({'gesture': gesture})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
