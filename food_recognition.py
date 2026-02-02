print("Program started...")

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# =====================
# SETTINGS
# =====================
DATASET_PATH = "dataset"
IMAGE_SIZE = 64

# Calories per 100 grams
CALORIES = {
    "apple": 52,
    "banana": 89,
    "orange": 47,
    "mix": 69
}

X = []
y = []
labels = []

# Allowed image extensions
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# =====================
# LOAD DATASET
# =====================
if not os.path.exists(DATASET_PATH):
    print("Dataset folder not found!")
    exit()

for fruit in os.listdir(DATASET_PATH):
    fruit_path = os.path.join(DATASET_PATH, fruit)
    if not os.path.isdir(fruit_path):
        continue

    print("Loading:", fruit)
    labels.append(fruit.lower())

    for img_name in os.listdir(fruit_path)[:100]:
        ext = os.path.splitext(img_name)[1].lower()
        if ext not in IMG_EXTENSIONS:
            continue  # skip non-image files

        try:
            img_path = os.path.join(fruit_path, img_name)
            img = Image.open(img_path).convert("RGBA")  # handle transparency
            img = img.convert("RGB")                     # remove alpha channel
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = np.array(img) / 255.0                 # normalize pixels
            X.append(img.flatten())
            y.append(labels.index(fruit.lower()))
        except Exception as e:
            print(f"Skipping {img_name}: {e}")
            continue

print("Images loaded successfully")

# =====================
# TRAIN MODEL
# =====================
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# =====================
# EVALUATION
# =====================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# =====================
# PREDICTION + CALORIES
# =====================
sample = X_test[0].reshape(1, -1)
predicted_label = model.predict(sample)[0]
food_name = labels[predicted_label]

print("Predicted Food:", food_name.capitalize())

# Ask user for weight
while True:
    try:
        weight = float(input(f"Enter the weight of {food_name} in grams: "))
        if weight <= 0:
            print("Please enter a positive number.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a number.")

# Calculate calories
calories = CALORIES[food_name] * (weight / 100)
print(f"Estimated Calories for {weight}g of {food_name.capitalize()}: {round(calories, 2)} kcal")

print("Program finished successfully")
