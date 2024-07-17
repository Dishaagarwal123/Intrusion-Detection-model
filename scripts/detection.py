import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('DNP3_Intrusion_Detection_Dataset\Training_Testing_Balanced_CSV_Files\Custom_DNP3_Parser\Custom_DNP3_Parser_Testing_Balanced.csv')
output = data.iloc[:, -1]
input_data = data.iloc[:, :-1]

correlation_matrix = input_data.corr()
correlation_threshold = 0.95
mask = (correlation_matrix.abs() >= correlation_threshold) & (correlation_matrix.abs() < 1.0)
features_to_remove = set()

for feature in correlation_matrix.columns:
    if feature not in features_to_remove:
        correlated_features = list(correlation_matrix.index[mask[feature]])
        correlated_features.remove(feature)
        features_to_remove.update(correlated_features)

features_to_remove = list(features_to_remove)

input_data_filtered = input_data.drop(columns=features_to_remove)

scaler = MinMaxScaler()
input_data_normalized = scaler.fit_transform(input_data_filtered)
input_data_normalized_df = pd.DataFrame(input_data_normalized, columns=input_data_filtered.columns)
print(input_data_normalized_df.head())

X_train, X_test, y_train, y_test = train_test_split(input_data_normalized_df, output, test_size=0.3, random_state=42)

svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the SVM model:", accuracy)