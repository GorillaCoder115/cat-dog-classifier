from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('dog_cat_model.h5')

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_filename = None
    if request.method == "POST":
        img_file = request.files['img']
        img_filename = img_file.filename
        img_path = os.path.join("static", img_filename)
        img_file.save(img_path)

        # Preprocess image
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        prediction = "Dog" if pred > 0.5 else "Cat"

    return render_template("index.html", prediction=prediction, img_filename=img_filename)

if __name__ == "__main__":
    app.run(debug=True)