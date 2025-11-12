import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv('../ipynb/fruit_classification_dataset.csv')
print(df.head())

# Encode categorical columns with separate LabelEncoders
shape_encoder = LabelEncoder()
color_encoder = LabelEncoder()
taste_encoder = LabelEncoder()
fruit_encoder = LabelEncoder()

df['shape_encoded'] = shape_encoder.fit_transform(df['shape'])
df['color_encoded'] = color_encoder.fit_transform(df['color'])
df['taste_encoded'] = taste_encoder.fit_transform(df['taste'])
df['fruit_name_encoded'] = fruit_encoder.fit_transform(df['fruit_name'])

print(df[['shape', 'shape_encoded', 'color', 'color_encoded', 
          'taste', 'taste_encoded', 'fruit_name', 'fruit_name_encoded']].head())

# Define features and target
X = df[['size (cm)', 'shape_encoded', 'weight (g)', 'avg_price (₹)',
        'color_encoded', 'taste_encoded']]
Y = df['fruit_name_encoded']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=42)

# ===============================
# Naive Bayes Classifier
# ===============================
nb = GaussianNB()
nb.fit(X_train, Y_train)
y_pred_nb = nb.predict(X_test)

print("\n=== Naive Bayes Metrics ===")
print("Accuracy:", accuracy_score(Y_test, y_pred_nb))
print("Precision:", precision_score(Y_test, y_pred_nb, average='macro'))
print("Recall:", recall_score(Y_test, y_pred_nb, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred_nb, average='macro'))
print("Classification Report:\n", classification_report(Y_test, y_pred_nb, target_names=fruit_encoder.classes_))

# ===============================
# K-Nearest Neighbors Classifier
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, Y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\n=== KNN Metrics ===")
print("Accuracy:", accuracy_score(Y_test, y_pred_knn))
print("Precision:", precision_score(Y_test, y_pred_knn, average='macro'))
print("Recall:", recall_score(Y_test, y_pred_knn, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred_knn, average='macro'))
print("Classification Report:\n", classification_report(Y_test, y_pred_knn, target_names=fruit_encoder.classes_))
