import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Step 1: Load and Audit the Training Data
data = pd.read_csv('features.csv')

# Check for missing values
if data.isnull().values.any():
    print("There are missing values in the dataset")
else:
    print("No missing values in the dataset")

# Assuming 'Class Label' is equivalent to 'Label' in the R code
data['Class Label'] = data['Class Label'].map({0: 'Good', 1: 'Bad'})

# One-hot encode categorical variables
# Replace ['Country'] with the actual categorical column names in your dataset
data = pd.get_dummies(data, columns=['Country'])

# Step 2: Train the Random Forest Model
X = data.drop('Class Label', axis=1)
y = data['Class Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

model = RandomForestClassifier(n_estimators=1500, max_features=10, random_state=123)
model.fit(X_train, y_train)

print(model)

# Step 3: Estimate Accuracy
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Confusion matrix for the entire data
predictions_all = model.predict(X)
print(confusion_matrix(y, predictions_all))


# Calculating additional metrics
total_rows = len(data)
print(f"Total number of rows in the dataset: {total_rows}")

# Sum of values in the confusion matrix
sum_confusion_values_all = np.sum(confusion_matrix(y, predictions_all))
print(f"Sum of the values in the confusion matrix: {sum_confusion_values_all}")

# Check if they're equal
if total_rows == sum_confusion_values_all:
    print("The total number of rows in the dataset and the sum of the values in the confusion matrix are equal.")
else:
    print("There is a discrepancy between the total number of rows in the dataset and the sum of the values in the confusion matrix.")

# Step 4: ROC Curve and AUC
probabilities = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label='Bad')
plt.plot(fpr, tpr, color='blue', lw=2)
plt.title("ROC Curve")
plt.show()
print("AUC:", roc_auc_score(y_test, probabilities))


# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=3)
print("Cross-validation scores:", cv_scores)

# Step 5: Feature Ranking
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
print("Top 10 Features:", X.columns[sorted_indices[:10]])

# RF Run-Time Test
positive_sample = X_test[y_test == 'Good'].iloc[0].values.reshape(1, -1)
negative_sample = X_test[y_test == 'Bad'].iloc[0].values.reshape(1, -1)
print("Positive Sample Prediction:", model.predict(positive_sample))
print("Negative Sample Prediction:", model.predict(negative_sample))

# Additional data checks
original_shape = data.shape
new_shape = X.shape
print("Original Data Shape:", original_shape)
print("New Data Shape:", new_shape)


