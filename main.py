import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from lime import lime_tabular

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

model = RandomForestClassifier(n_estimators=1500, max_features=10, random_state=123, oob_score=True)
model.fit(X_train, y_train)

print(model)

# Step 3: Estimate Accuracy
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Confusion matrix for the entire data
predictions_all = model.predict(X)
print(confusion_matrix(y, predictions_all))

# OOB Score
print(f"OOB Score: {model.oob_score_}")

# Plotting OOB score
plt.figure(figsize=(7, 5))
plt.barh(['OOB Score'], [model.oob_score_], color='skyblue')
plt.xlabel('Score')
plt.title('Random Forest OOB Score')
plt.xlim(0, 1)  # OOB score is between 0 and 1
plt.show()

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

# Create a LimeTabularExplainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Good', 'Bad'],
    mode='classification'
)

# Explain a single prediction from the test set
i = 10  # Index of the sample in your test set
sample = X_test.iloc[i].astype(float)
exp = explainer.explain_instance(sample, model.predict_proba, num_features=10)

# Display LIME explanation for this sample in console
print(exp.as_list())

# Remember to adjust the index `i` to the specific instance you want to explain
# Display the LIME explanation in the console
# exp.as_list() provides a list of explanations, the following line prints them
for feature, importance in exp.as_list():
    print(f"{feature}: {importance}")

# Optionally, save the LIME explanation to a file (commented out here)
# with open('lime_explanation.txt', 'w') as file:
#     for feature, importance in exp.as_list():
#         file.write(f"{feature}: {importance}\n")

# Visualization (commented out because this doesn't work in a text-based environment)
# If running in Jupyter Notebook, you can uncomment the following line
# exp.show_in_notebook(show_all=False)

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, predictions, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot Feature Importances
plt.figure(figsize=(40, 24))
sns.barplot(x=importances[sorted_indices[:10]], y=X.columns[sorted_indices[:10]])
plt.title('Top 10 Feature Importances')
plt.show()




# Step 4: ROC Curve and AUC
probabilities = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label='Bad')

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_score(y_test, probabilities)))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



###############################################
# Precision Recall Curve
###############################################
# Convert 'Good' to 0 and 'Bad' to 1
y_test_binary = y_test.map({'Good': 0, 'Bad': 1})
precision, recall, _ = precision_recall_curve(y_test_binary, probabilities)


plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()

###############################################
# Feature Importance Heatmap (Top 10)
###############################################

import seaborn as sns

# Assuming 'model' is your trained RandomForest model and 'X_train' is your training data
importances = model.feature_importances_
feature_names = X_train.columns

# Create a dataframe of feature importances
df_feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
df_feature_importance = df_feature_importance.sort_values('importance', ascending=False)

# Plotting
plt.figure(figsize=(30, 40))
sns.heatmap(df_feature_importance.set_index('feature').T, cmap='viridis', annot=True)
plt.title('Feature Importance Heatmap')
plt.show()


###############################################
# Confusion Matrix Visualization
###############################################

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming 'y_test' contains true labels and 'predictions' contain model predictions
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()



# Create a LimeTabularExplainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Good', 'Bad'],
    mode='classification'
)

# Choose a sample to explain
i = 10  # Index of the sample in your test set
sample = X_test.iloc[i]

# Generate LIME explanation for this sample
exp = explainer.explain_instance(sample.values, model.predict_proba, num_features=10)

# Show the explanation
exp.show_in_notebook(show_table=True)

################################################
################################################
############        OUTPUT          ############
################################################
################################################


# /Users/dylanburns/PycharmProjects/AIX_proj/venv/bin/python /Users/dylanburns/PycharmProjects/AIX_proj/main.py
# No missing values in the dataset
# RandomForestClassifier(max_features=10, n_estimators=1500, oob_score=True,
#                        random_state=123)
#               precision    recall  f1-score   support
#
#          Bad       0.79      0.74      0.77        31
#         Good       0.72      0.78      0.75        27
#
#     accuracy                           0.76        58
#    macro avg       0.76      0.76      0.76        58
# weighted avg       0.76      0.76      0.76        58
#
# [[87  8]
#  [ 6 90]]
# OOB Score: 0.6842105263157895
# Total number of rows in the dataset: 191
# Sum of the values in the confusion matrix: 191
# The total number of rows in the dataset and the sum of the values in the confusion matrix are equal.
# AUC: 0.8602150537634409
# Cross-validation scores: [0.828125   0.671875   0.71428571]
# Top 10 Features: Index(['1.4.1_SP_ACS_BSRVSAN__ALLAREA__2019', '10.7.4_SM_POP_REFG_OR____2019',
#        '8.4.2_EN_MAT_DOMCMPG____2019', '1.4.1_SP_ACS_BSRVH2O__ALLAREA__2019',
#        '3.2.1_SH_DYN_MORT_BOTHSEX__<5Y_2019', '9.2.1_NV_IND_MANFPC____2019',
#        '3.2.1_SH_DYN_MORT_MALE__<5Y_2019',
#        '3.2.1_SH_DYN_MORT_FEMALE__<5Y_2019',
#        '3.2.1_SH_DYN_IMRT_BOTHSEX__<1Y_2019', '17.13.1_PA_NUS_ATLS____2019'],
#       dtype='object')
# Positive Sample Prediction: ['Good']
# Negative Sample Prediction: ['Good']
# Original Data Shape: (191, 322)
# New Data Shape: (191, 321)

# [('Country_Czechia > 0.00', 0.07567567990545487), ('1.4.1_SP_ACS_BSRVSAN__ALLAREA__2019 > 98.05', 0.027543850388946283), ('Country_Hungary <= 0.00', 0.0245271298332754), ('3.2.1_SH_DYN_MORT_MALE__<5Y_2019 <= 6.20', 0.019654076941809987), ('Country_Paraguay <= 0.00', -0.019553832522414414), ('3.2.1_SH_DYN_MORT_FEMALE__<5Y_2019 <= 5.10', 0.017317037892084684), ('3.2.1_SH_DYN_MORT_BOTHSEX__<5Y_2019 <= 5.70', 0.01716750453814799), ('9.2.1_NV_IND_MANFPC____2019 > 1879.98', 0.017066995343343945), ('3.2.1_SH_DYN_IMRT_FEMALE__<1Y_2019 <= 4.20', 0.015749065621382237), ('Country_Estonia <= 0.00', -0.011321357479535563)]
# Country_Czechia > 0.00: 0.07567567990545487
# 1.4.1_SP_ACS_BSRVSAN__ALLAREA__2019 > 98.05: 0.027543850388946283
# Country_Hungary <= 0.00: 0.0245271298332754
# 3.2.1_SH_DYN_MORT_MALE__<5Y_2019 <= 6.20: 0.019654076941809987
# Country_Paraguay <= 0.00: -0.019553832522414414
# 3.2.1_SH_DYN_MORT_FEMALE__<5Y_2019 <= 5.10: 0.017317037892084684
# 3.2.1_SH_DYN_MORT_BOTHSEX__<5Y_2019 <= 5.70: 0.01716750453814799
# 9.2.1_NV_IND_MANFPC____2019 > 1879.98: 0.017066995343343945
# 3.2.1_SH_DYN_IMRT_FEMALE__<1Y_2019 <= 4.20: 0.015749065621382237
# Country_Estonia <= 0.00: -0.011321357479535563
#
# Process finished with exit code 1