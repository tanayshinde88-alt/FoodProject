Food Recognition and Calorie Estimation using SVM (Python)

This project builds a machine learning system that can recognize different food items from images and estimate their calorie content based on the weight provided by the user. It uses a Support Vector Machine (SVM) for classification and a simple calorie lookup for estimation.

The aim is to combine image classification with a practical nutrition use case.

+ Repository Structure
food-recognition-calorie-estimator/
│
├── food_recognition.py      → Main Python script
├── dataset/                → Dataset folder
│   ├── apple/              → Apple images
│   ├── banana/             → Banana images
│   ├── orange/             → Orange images
│   └── mix/                → Mixed fruit images
└── README.md               → Project documentation


Note:
The full dataset is not included due to size limits.
You should place your own images inside the dataset/ folder following the structure above.

+ Project Overview

Loads food images from folders

Resizes images to 64 × 64

Normalizes pixel values

Converts images into numerical vectors

Splits data into training and testing sets

Trains a linear SVM classifier

Predicts food type from an image

Asks user for food weight

Estimates calories based on known values

+ Technologies Used

Python

NumPy

Pillow (PIL)

Scikit-learn

+ How to Run the Project

Install required libraries:

pip install numpy pillow scikit-learn


Make sure your folder structure looks like this:

dataset/
  apple/
  banana/
  orange/
  mix/


Run the script:

python food_recognition.py


The program will:

Train the model

Show model accuracy

Predict a food item

Ask for weight in grams

Display estimated calories

+ Output

Console output showing:

Model accuracy

Predicted food name

Estimated calories

Example:

Model Accuracy: 88.45 %
Predicted Food: Banana
Enter the weight of banana in grams: 150
Estimated Calories for 150g of Banana: 133.5 kcal

+ Use Case

This project can help with:

Learning food image classification

Understanding ML pipelines for images

Building nutrition-related applications

+ Future Improvements

Add more food categories

Use deep learning (CNN)

Add GUI or web interface

Use real-time camera input
