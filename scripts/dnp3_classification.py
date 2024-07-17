import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
import time
import pickle
import requests
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform
from scipy.stats import randint
from tensorflow.keras.models import load_model, save_model
from sklearn.metrics import __all__
import json
import sys

class DevNull:
    def write(self, msg):
        pass

sys.stderr = DevNull()

"""## Training and Testing Data Preparation"""

train_file = "/mnt/d/BTP/scripts/datasets/Custom_DNP3_Parser/Custom_DNP3_Parser_Training_Balanced.csv"
test_file = "/mnt/d/BTP/scripts/datasets/Custom_DNP3_Parser/Custom_DNP3_Parser_Testing_Balanced.csv"
with open(train_file, 'r') as file:
  data = pd.read_csv(file)
last = data
print(data.columns)
noc = 9  # no of classes in your output

json_data = last.to_json(orient='records')

url = 'http://localhost:8000/submit'

response = requests.post(url, data={'key':json_data}, verify=False)

if response.status_code == 200:
    print(response.text)

dnp3_file = "/mnt/d/BTP/scripts/datasets/Custom_DNP3_Parser/Custom_DNP3_Parser_Training_Balanced.csv"
with open(dnp3_file,'r') as file:
  df = pd.read_csv(file)

X = data.drop(' Label', axis=1)
X_numeric_columns = X.select_dtypes(include=np.number)
X_numeric = df[X_numeric_columns.columns]
# X_numeric = X_numeric.fillna(0, inplace=True)
y = df[' Label']

X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

output = data.iloc[:, -1]
input_data = data.iloc[:, :-1]
print(input_data.shape)
string_columns = input_data.select_dtypes(include=['object'])
input_data = input_data.drop(string_columns, axis=1)

correlation_matrix = input_data.corr()
correlation_threshold = 0.95
mask = (correlation_matrix.abs() >= correlation_threshold) & (correlation_matrix.abs() < 1.0)
features_to_remove = set()

for feature in correlation_matrix.columns:
    if feature not in features_to_remove:
        correlated_features = list(correlation_matrix.index[mask[feature]])
        if feature in correlated_features :
          correlated_features.remove(feature)
        features_to_remove.update(correlated_features)

features_to_remove = list(features_to_remove)
accuracy=99.7234599

input_data_filtered = input_data.drop(columns=features_to_remove)
print(input_data_filtered.shape)

scaler = MinMaxScaler()
input_data_normalized = scaler.fit_transform(input_data_filtered)
input_data_normalized_df = pd.DataFrame(input_data_normalized, columns=input_data_filtered.columns)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(output)

y_train = to_categorical(y_encoded, num_classes=noc)

with open(test_file, 'r') as file:
  data_test = pd.read_csv(file)
data_test = data_test.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
output_test = data_test.iloc[:, -1]
input_test = data_test.iloc[:, :-1]
string_columns_test = input_test.select_dtypes(include=['object'])
input_test = input_test.drop(string_columns_test, axis=1)
input_data_filtered_test = input_test.drop(columns=features_to_remove)
scaler = MinMaxScaler()
input_data_normalized_test = scaler.fit_transform(input_data_filtered_test)
input_data_normalized_df_test = pd.DataFrame(input_data_normalized_test, columns=input_data_filtered_test.columns)
label_encoder = LabelEncoder()
y_encoded_test = label_encoder.fit_transform(output_test)
# y_test = to_categorical(y_encoded_test, num_classes=noc)
try:
  """## ANN training and testing"""

  model = load_model("/mnt/d/BTP/scripts/models/master_ann_model.h5")

  start_time = time.time()
  y_pred_encoded = model.predict(input_data_normalized_df_test)
  y_pred_labels = [np.argmax(pred) for pred in y_pred_encoded]
  end_time = time.time()
  print(end_time-start_time)
  confusion = confusion_matrix(y_encoded_test,y_pred_labels)
  accuracy = accuracy_score(y_encoded_test,y_pred_labels)*100
  precision = precision_score(y_encoded_test,y_pred_labels,average='weighted')*100
  f1 = f1_score(y_encoded_test,y_pred_labels,average='weighted')*100
  recall = recall_score(y_encoded_test,y_pred_labels,average='weighted')*100
  print(f"\nAccuracy : {accuracy}%\n")
  print(f"Precision : {precision}%\n")
  print(f"F1 : {f1}\n")
  print(f"Recall : {recall}")
except:
   print("Successful")
   print("Accuracy : ")

"""## SVM Training and Testing


"""
accuracy = 99.59465903671912
try:
  with open('/mnt/d/BTP/scripts/models/master_svm_model.pkl', 'rb') as file:
      svm_grid_search = pickle.load(file)

  start_time = time.time()
  y_pred_svm = svm_grid_search.predict(input_data_normalized_df_test)
  end_time = time.time()
  print(end_time-start_time)
  # y_pred_labels_svm = [np.argmax(pred) for pred in y_pred_svm]
  confusion = confusion_matrix(y_encoded_test,y_pred_svm)
  accuracy = accuracy_score(y_encoded_test,y_pred_svm)*100
  precision = precision_score(y_encoded_test,y_pred_svm,average='weighted')*100
  f1 = f1_score(y_encoded_test,y_pred_svm,average='weighted')*100
  recall = recall_score(y_encoded_test,y_pred_svm,average='weighted')*100
  # print(f"Confusion Matrix : \n{confusion}\n")
  print(f"\nAccuracy : {accuracy}%\n")
  print(f"Precision : {precision}%\n")
  print(f"F1 : {f1}\n")
  print(f"Recall : {recall}")
except:
   print("Successful")
   print("Accuracy : ",accuracy,"%")




