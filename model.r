# -----------------------------------------------------------
# Install and Load Necessary Libraries
# -----------------------------------------------------------
install.packages(c("randomForest", "caret", "pROC"))
library(randomForest)
library(caret)
library(pROC)


# -----------------------------------------------------------
# 1. Import and Audit the Training Data
# -----------------------------------------------------------
mydata <- read.csv("https://raw.githubusercontent.com/cjtrowbridge/Sustainability-AI/main/data/features.csv")

# Check for missing values
if (any(is.na(mydata))) {
    print("There are missing values in the dataset")
} else {
    print("No missing values in the dataset")
}

#Assign class names
mydata$Label <- as.character(mydata$Label)
mydata$Label[mydata$Label == "0"] <- "Good"
mydata$Label[mydata$Label == "1"] <- "Bad"
mydata$Label <- as.factor(mydata$Label)

# Convert the 'Label' column to factor as it's the target variable
mydata$Label <- as.factor(mydata$Label)

# Splitting the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(mydata$Label, p = 0.8, list = FALSE)
dataTRAIN <- mydata[trainIndex,]
dataTEST <- mydata[-trainIndex,]


# -----------------------------------------------------------
# 2. Train RF
# -----------------------------------------------------------

# Train the model with the specified ntree and mtry values
model <- randomForest(
	formula = Label ~ .,
	data = dataTRAIN,
	ntree = 1500,
	mtry = 10,
	importance = TRUE
)


# Print the results
print(model)

# -----------------------------------------------------------
# Accuracy Estimation
# -----------------------------------------------------------

# Predict on test data and generate confusion matrix
predictions <- predict(model, dataTEST)
predictions_all <- predict(model, mydata)

confMatrix_all <- confusionMatrix(predictions_all, mydata$Label)
print(confMatrix_all)

# Calculate the total number of rows in the dataset
total_rows <- nrow(mydata)
print(paste("Total number of rows in the dataset:", total_rows))

# Extract the values from the new confusion matrix
confusion_values_all <- as.numeric(confMatrix_all$table)

# Sum the values from the new confusion matrix
sum_confusion_values_all <- sum(confusion_values_all)
print(paste("Sum of the values in the confusion matrix:", sum_confusion_values_all))

# Check if they're equal
if (total_rows == sum_confusion_values_all) {
  print("The total number of rows in the dataset and the sum of the values in the confusion matrix are equal.")
} else {
  print("There is a discrepancy between the total number of rows in the dataset and the sum of the values in the confusion matrix.")
}

# Extract the F1 Score
f1_score <- confMatrix_all$byClass["F1"]
print(paste("F1 Score:", f1_score))

# Extract Sensitivity (True Positive Rate)
sensitivity <- confMatrix_all$byClass["Sensitivity"]
print(paste("Sensitivity (True Positive Rate):", sensitivity))

# Extract Specificity (True Negative Rate)
specificity <- confMatrix_all$byClass["Specificity"]
print(paste("Specificity (True Negative Rate):", specificity))

# Compare with OOB error
oobError <- 1 - model$err.rate[nrow(model$err.rate), "OOB"]
print(paste("OOB Error:", oobError))

# Predict probabilities for the positive class
probabilities <- predict(model, dataTEST, type = "prob")[,2]

# Compute and plot ROC curve
roc_curve <- roc(dataTEST$Label, probabilities)
plot(roc_curve, main="ROC Curve", col="blue", lwd=2)
abline(h=0, v=1, col="gray", lty=2)

# Compute and print AUC
auc_val <- auc(roc_curve)
print(paste("AUC:", auc_val))


# -----------------------------------------------------------
# R Cross Validation
# -----------------------------------------------------------

# Define the controls
control <- trainControl(method="cv", number=3)

# Train the model using 3-fold cross-validation with a fixed mtry and ntree
cv_model <- train(
  Label ~ .,
  data=mydata,
  method="rf",
  trControl=control,
  tuneGrid=data.frame(.mtry=10),
  ntree=1000
)

# Print the results
print(cv_model)


# -----------------------------------------------------------
# Feature Importances
# -----------------------------------------------------------
# Extract feature importance
featureImportance <- randomForest::importance(model)

# Sort the featureImportance data frame by MeanDecreaseAccuracy column in descending order
sortedFeatureImportance <- featureImportance[order(-featureImportance[, "MeanDecreaseAccuracy"]), ]

# Print the top features
print(sortedFeatureImportance)


# -----------------------------------------------------------
# Save Feature Importances to CSV
# -----------------------------------------------------------

# Specify the file path
output_file_path <- "C:/Users/CJ/Drive/School/2023/03 - Fall/2 - CSC 859 - AI Explainability and Ethics/AI Project/sdgs/significance.csv"

# Write the sortedFeatureImportance data frame to a CSV file
write.csv(sortedFeatureImportance, file = output_file_path, row.names = TRUE)


# -----------------------------------------------------------
# Predictions for 2 Random Samples
# -----------------------------------------------------------
randomSamples <- mydata[sample(nrow(mydata), 2), ]
predictionsRandomSamples <- predict(model, randomSamples)
print(predictionsRandomSamples)
