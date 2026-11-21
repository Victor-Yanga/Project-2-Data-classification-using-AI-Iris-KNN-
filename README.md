Project 2: Data Classification Using AI (Iris, KNN)

DecodeLabs AI Internship, Batch 2026. Author: Victor Usoh.

A supervised-learning pipeline that classifies iris flowers into three species (setosa, versicolor, virginica) from four measurements.

What it does

1. Loads and explores the Iris dataset (150 samples, 4 features, 3 balanced classes).
2. Splits the data 80/20 into training and testing sets (shuffled and stratified, fixed random seed).
3. Scales the features with StandardScaler, fitted on the training data only to avoid data leakage.
4. Compares several values of K, then trains a K-Nearest Neighbors model with K = 5.
5. Evaluates with accuracy, a confusion matrix and precision / recall / F1 scores.
6. Predicts the species of a new flower the model has not seen.

Results (K = 5)

• Accuracy: 93.3% on 30 test samples
• Setosa was classified perfectly. Two virginica flowers were mistaken for versicolor, the two species whose measurements overlap most.

Run it

pip install scikit-learn pandas matplotlib
python iris_knn_classifier.py 

The script prints the results and saves confusion_matrix.png.

Limitations and next steps

• The test set is small (30 samples), so one or two mistakes move the accuracy a lot.
• K was compared on the test set for illustration. A stricter approach would choose K with cross-validation.
• Next: try other algorithms (decision tree, logistic regression) and compare them.