from flask import Flask, render_template, request
import pickle
import cv2
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]

        if not file:
            return render_template("index.html", result="No image uploaded")

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        img = img.flatten().reshape(1, -1)
        img = img / 255.0

        pred = model.predict(img)[0]
        result = "Tumor Detected" if pred == 1 else "No Tumor"

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
    