# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset (from a CSV file or online source)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Preprocess the data
# Select relevant features and drop rows with missing values
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()

# Encode categorical data (e.g., 'Sex')
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])  # Male: 1, Female: 0

# Split the dataset into features (X) and target (y)
X = data[['Pclass', 'Sex', 'Age', 'Fare']]  # Features
y = data['Survived']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Decision Tree Classification - Titanic Dataset")
print("---------------------------------------------")
print(f"Accuracy: {accuracy:.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Display feature importance
feature_importance = pd.DataFrame({
    'Feature': ['Pclass', 'Sex', 'Age', 'Fare'],
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)
