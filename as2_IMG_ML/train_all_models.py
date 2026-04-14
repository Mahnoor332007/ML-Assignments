import os
import cv2
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

data = []
labels = []

dataset_path = "brain_dataset"
categories = ["yes", "no"]

# Load images
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

X = np.array(data) / 255.0
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

metrics_data = {}

# -----------------------------
# KNN
# -----------------------------
knn_params = {
    "n_neighbors": [3, 5, 7],
    "weights": ["uniform", "distance"],
    "p": [1, 2]
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=3,
    scoring="f1",
    n_jobs=-1
)
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_

knn_pred = best_knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)
knn_cm = confusion_matrix(y_test, knn_pred).tolist()

print("\n" + "=" * 60)
print("Model: KNN")
print("=" * 60)
print("Best Parameters:", knn_grid.best_params_)
print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)
print("Confusion Matrix:\n", knn_cm)

with open("knn_model.pkl", "wb") as f:
    pickle.dump(best_knn, f)

metrics_data["KNN"] = {
    "best_params": knn_grid.best_params_,
    "accuracy": round(knn_accuracy, 4),
    "precision": round(knn_precision, 4),
    "recall": round(knn_recall, 4),
    "f1_score": round(knn_f1, 4),
    "confusion_matrix": knn_cm
}

# -----------------------------
# Decision Tree
# -----------------------------
dt_params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=3,
    scoring="f1",
    n_jobs=-1
)
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_

dt_pred = best_dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_cm = confusion_matrix(y_test, dt_pred).tolist()

print("\n" + "=" * 60)
print("Model: Decision Tree")
print("=" * 60)
print("Best Parameters:", dt_grid.best_params_)
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)
print("Confusion Matrix:\n", dt_cm)

with open("dt_model.pkl", "wb") as f:
    pickle.dump(best_dt, f)

metrics_data["Decision Tree"] = {
    "best_params": dt_grid.best_params_,
    "accuracy": round(dt_accuracy, 4),
    "precision": round(dt_precision, 4),
    "recall": round(dt_recall, 4),
    "f1_score": round(dt_f1, 4),
    "confusion_matrix": dt_cm
}

# -----------------------------
# Naive Bayes
# -----------------------------
nb_params = {
    "var_smoothing": [1e-09, 1e-08, 1e-07]
}

nb_grid = GridSearchCV(
    GaussianNB(),
    nb_params,
    cv=3,
    scoring="f1",
    n_jobs=-1
)
nb_grid.fit(X_train, y_train)
best_nb = nb_grid.best_estimator_

nb_pred = best_nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)
nb_cm = confusion_matrix(y_test, nb_pred).tolist()

print("\n" + "=" * 60)
print("Model: Naive Bayes")
print("=" * 60)
print("Best Parameters:", nb_grid.best_params_)
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)
print("Confusion Matrix:\n", nb_cm)

with open("nb_model.pkl", "wb") as f:
    pickle.dump(best_nb, f)

metrics_data["Naive Bayes"] = {
    "best_params": nb_grid.best_params_,
    "accuracy": round(nb_accuracy, 4),
    "precision": round(nb_precision, 4),
    "recall": round(nb_recall, 4),
    "f1_score": round(nb_f1, 4),
    "confusion_matrix": nb_cm
}

# -----------------------------
# Best Model Selection
# -----------------------------
if knn_f1 > dt_f1 and knn_f1 > nb_f1:
    best_model = best_knn
    best_name = "KNN"
    best_f1 = knn_f1
elif dt_f1 > nb_f1:
    best_model = best_dt
    best_name = "Decision Tree"
    best_f1 = dt_f1
else:
    best_model = best_nb
    best_name = "Naive Bayes"
    best_f1 = nb_f1

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("best_model_name.txt", "w") as f:
    f.write(best_name)

metrics_data["Best Model"] = {
    "name": best_name,
    "f1_score": round(best_f1, 4)
}

with open("metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\n" + "=" * 60)
print("FINAL BEST MODEL:", best_name)
print("BEST MODEL F1 SCORE:", round(best_f1, 4))
print("=" * 60)

print("\nAll models and metrics saved successfully!")

# -----------------------------
# Combined Confusion Matrix Plot
# -----------------------------
class_names = ["No Tumor", "Tumor"]

plot_configs = [
    (np.array(knn_cm), "KNN", "Blues", knn_accuracy),
    (np.array(dt_cm),  "Decision Tree", "Greens", dt_accuracy),
    (np.array(nb_cm),  "Naive Bayes", "Oranges", nb_accuracy),
]

# GridSpec: 2 rows — top row for heatmaps, bottom row for table
# height_ratios gives the heatmap row more space than the table row
fig = plt.figure(figsize=(22, 14))
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 3, figure=fig,
              height_ratios=[1.6, 1],
              hspace=0.55,   # vertical gap between rows
              wspace=0.4)    # horizontal gap between columns

# --- Row 0: Confusion matrix heatmaps ---
for i, (cm, title, cmap, acc) in enumerate(plot_configs):
    ax = fig.add_subplot(gs[0, i])
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, annot_kws={"size": 15},
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_title(f"{title}\n(Accuracy: {acc * 100:.2f}%)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("Actual Label", fontsize=11, labelpad=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10, rotation=0)

# --- Row 1: Accuracy comparison table (spans all 3 columns) ---
ax_table = fig.add_subplot(gs[1, :])
ax_table.axis("off")

table_data = [
    ["KNN",           f"{knn_accuracy * 100:.2f}%",  f"{knn_precision:.4f}",  f"{knn_recall:.4f}",  f"{knn_f1:.4f}"],
    ["Decision Tree", f"{dt_accuracy * 100:.2f}%",   f"{dt_precision:.4f}",   f"{dt_recall:.4f}",   f"{dt_f1:.4f}"],
    ["Naive Bayes",   f"{nb_accuracy * 100:.2f}%",   f"{nb_precision:.4f}",   f"{nb_recall:.4f}",   f"{nb_f1:.4f}"],
]
col_labels = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]

table = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
    bbox=[0.05, 0.05, 0.9, 0.8]  # [left, bottom, width, height] within the subplot
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.4)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#4C72B0")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Highlight best model row in green
best_row = ["KNN", "Decision Tree", "Naive Bayes"].index(best_name) + 1
for j in range(len(col_labels)):
    table[best_row, j].set_facecolor("#d4edda")

ax_table.set_title("Model Accuracy Comparison", fontsize=13,
                   fontweight="bold", pad=16, loc="center")

# Main title — y pushed above the top of the figure
fig.suptitle("Confusion Matrices & Accuracy Table — All Models",
             fontsize=17, fontweight="bold", y=0.98)

plt.savefig("all_confusion_matrices.png", bbox_inches="tight", dpi=150)
plt.show()
