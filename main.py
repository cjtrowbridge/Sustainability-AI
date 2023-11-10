import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, \
    precision_score, roc_curve, roc_auc_score, precision_recall_curve
import seaborn as sns

# Step 1: Load and Audit the Training Data
data = pd.read_csv('i1_positive.csv')
# Audit the dataset if required and perform any preprocessing.

# Step 2: Train the Random Forest Model
# Split the data into features (X) and the target variable (y)
X = data.drop('Label', axis=1)
y = data['Label']

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the hyperparameters and their ranges for grid search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30, None],
    'max_features': ['sqrt'],
}

# Create the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=0)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the model to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_rf_classifier = grid_search.best_estimator_

# Train the best model
best_rf_classifier.fit(X_train, y_train)

# Step 3: Estimate Accuracy
# Make predictions on the test set
y_pred = best_rf_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Calculate and display Precision, Recall, and F1 Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Step 4: Feature Ranking
# You can extract feature importances from the trained model
feature_importances = best_rf_classifier.feature_importances_

# Sort the features by importance
sorted_feature_indices = feature_importances.argsort()[::-1]
top_10_feature_indices = sorted_feature_indices[:10]

# Get the feature names for the top 10 features
top_10_features = X.columns[top_10_feature_indices]
print("Top 10 Features:", top_10_features)

# # Step 5: RF Run-Time Test
# # Choose two samples from the dataset for runtime testing
# sample1 = X_test.iloc[0].values.reshape(1, -1)
# sample2 = X_test.iloc[1].values.reshape(1, -1)
#
# # Predict the class of the samples using the trained RF model
# sample1_prediction = best_rf_classifier.predict(sample1)
# sample2_prediction = best_rf_classifier.predict(sample2)
#
# print("Sample 1 Prediction:", sample1_prediction)
# print("Sample 2 Prediction:", sample2_prediction)
# Select one positive and one negative sample from the training data
positive_sample = X_train[y_train == 1].iloc[0].values.reshape(1, -1)
negative_sample = X_train[y_train == 0].iloc[0].values.reshape(1, -1)

# Predict the class of the samples using the trained RF model
positive_sample_prediction = best_rf_classifier.predict(positive_sample)
negative_sample_prediction = best_rf_classifier.predict(negative_sample)

# Display the predictions
print("Positive Sample Prediction:", positive_sample_prediction)
print("Negative Sample Prediction:", negative_sample_prediction)


# Calculate ROC curve
y_prob = best_rf_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='best')
plt.show()

###############################################
# Feature Importance Heatmap (Top 10)
###############################################

# Plot feature importance (Top 10)
plt.figure(figsize=(10, 6))
plt.barh(range(10), feature_importances[top_10_feature_indices])
plt.yticks(range(10), top_10_features)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Top 10 Feature Importance')
plt.show()

###############################################
# Confusion Matrix Visualization
###############################################

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()






















