import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load the dataset
df = pd.read_csv("heart.csv")

# Display dataset
print(df.head())
print(df.shape)

# Separate input features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Display X and y
print(X.head())
print(y.head())

# Store encoders for each categorical column
encoders = {}

# Find all categorical (text) columns
categorical_columns = X.select_dtypes(include=["object"]).columns

# Encode each categorical column and save its encoder
for column in categorical_columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])
    encoders[column] = encoder
    
    # Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(comparison.head(10))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Display detailed evaluation metrics
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

tn, fp, fn, tp = cm.ravel()

print("True Negative:", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive:", tp)

# Save the trained model
joblib.dump(model, "heart_model.pkl")

# Save column names
joblib.dump(X.columns.tolist(), "columns.pkl")
joblib.dump(encoders, "encoders.pkl")
print("Model and columns saved successfully!")

