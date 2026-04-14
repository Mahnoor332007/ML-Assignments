import os
import cv2
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = []
labels = []

dataset_path = "dataset"   # ⚠️ agar folder ka naam brain_dataset hai to change karna

categories = ["yes", "no"]

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = 1 if category == "yes" else 0

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img = img.flatten()

            data.append(img)
            labels.append(label)
        except:
            pass

X = np.array(data)
y = np.array(labels)

# Normalize
X = X / 255.0

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
