from flask import Flask, render_template, request
import pickle
import cv2
import numpy as np
import json

app = Flask(__name__)

# Load individual models
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

# Load best model
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("best_model_name.txt", "r") as f:
    best_model_name = f.read().strip()

# Load metrics
with open("metrics.json", "r") as f:
    metrics = json.load(f)

@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]

        if not file:
            return render_template(
                "index.html",
                knn_result="No image uploaded",
                dt_result="",
                nb_result="",
                best_result="",
                metrics=metrics
            )

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return render_template(
                "index.html",
                knn_result="Invalid image file",
                dt_result="",
                nb_result="",
                best_result="",
                metrics=metrics
            )

        img = cv2.resize(img, (64, 64))
        img = img.flatten().reshape(1, -1)
        img = img / 255.0

        knn_pred = knn_model.predict(img)[0]
        dt_pred = dt_model.predict(img)[0]
        nb_pred = nb_model.predict(img)[0]
        best_pred = best_model.predict(img)[0]

        knn_result = "KNN Prediction: Tumor Detected" if knn_pred == 1 else "KNN Prediction: No Tumor"
        dt_result = "Decision Tree Prediction: Tumor Detected" if dt_pred == 1 else "Decision Tree Prediction: No Tumor"
        nb_result = "Naive Bayes Prediction: Tumor Detected" if nb_pred == 1 else "Naive Bayes Prediction: No Tumor"
        best_result = f"Best Model ({best_model_name}): " + ("Tumor Detected" if best_pred == 1 else "No Tumor")

        return render_template(
            "index.html",
            knn_result=knn_result,
            dt_result=dt_result,
            nb_result=nb_result,
            best_result=best_result,
            metrics=metrics
        )

    except Exception as e:
        return render_template(
            "index.html",
            knn_result=f"Error: {str(e)}",
            dt_result="",
            nb_result="",
            best_result="",
            metrics=metrics
        )

if __name__ == "__main__":
    app.run(debug=True)
    