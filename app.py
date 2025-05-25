import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Paths & config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths should be relative to BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "all_models", "mobilenetv2_waste_classifier_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "webapp", "static", "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once when server starts
model = load_model(MODEL_PATH)
class_names = ['glass', 'metal', 'organic', 'paper', 'plastic']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Prepare and predict
            img_ready = prepare_image(save_path)
            preds = model.predict(img_ready)
            predicted_class = class_names[np.argmax(preds)]

            # URL for the uploaded image (to show in modal)
            image_url = url_for('static', filename='uploads/' + filename)

            return render_template('index.html', prediction=predicted_class, image_url=image_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
