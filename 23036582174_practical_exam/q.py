import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('loan_approval.csv')
# print(df.head())

le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])
df['city'] = le.fit_transform(df['city'])
df['loan_approved'] = le.fit_transform(df['loan_approved'])
print(df.head())

x = df.drop('loan_approved', axis=1)
y = df['loan_approved']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# K-NN Model
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_pred_decode = le.inverse_transform(knn_pred)
y_test_decode = le.inverse_transform(y_test)
print(knn_pred_decode)
print(classification_report(y_test_decode, knn_pred_decode))
print("KNN 6-Fold Accuracy:")
print(cross_val_score(knn, x, y, cv=6))


# Decision Tree Model
# dt = DecisionTreeClassifier(
#     criterion='gini',       
#     max_depth=None                  
# )

# dt.fit(X_train, y_train)

# dt_pred = dt.predict(X_test)
# dt_pred_decode = le.inverse_transform(dt_pred)
# print(dt_pred_decode)
# print(classification_report(y_test, dt_pred_decode))
# print("Decision Tree 6-Fold Accuracy:")
# print(cross_val_score(dt, x, y, cv=6))