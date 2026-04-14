from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load model
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form["text"]

    if not user_text.strip():
        return render_template("index.html", result="Please enter text")

    result = classifier(user_text)[0]

    label = result["label"]
    score = result["score"]

    output = f"{label} (Confidence: {score:.2f})"

    return render_template("index.html", result=output, text=user_text)

if __name__ == "__main__":
    app.run(debug=True)