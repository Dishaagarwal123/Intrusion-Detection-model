import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, save_model
import requests
import json



data_file = "/mnt/d/BTP/scripts/datasets/packet_data.xlsx"

dataset = pd.read_excel(data_file)

X = dataset.iloc[:, :-1]
X_numeric = X.select_dtypes(exclude=['object'])

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_numeric)

y_name = dataset.iloc[:, -1]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_name)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# model = Sequential()
# model.add(Input(shape = (X_train.shape[1],)))
# for i in range(1,11) :
#   model.add(Dense(90, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train , epochs=100, batch_size=32,verbose=0)
# save_model(model,"/mnt/d/BTP/scripts/models/master_model.h5")

model = load_model("/mnt/d/BTP/scripts/models/master_model.h5")

loss, accuracy = model.evaluate(X_normalized, y)

print(f"Test Accuracy: {accuracy}")
percentage_accuracy = accuracy * 100
print(f"Percentage Accuracy: {percentage_accuracy:.2f}%")

# loss, accuracy = model.evaluate(X_normalized, y)

# percentage_accuracy = accuracy * 100
# print(f"Percentage Accuracy: {percentage_accuracy:.2f}%")

predictions = model.predict(X_normalized)

predicted_labels = label_encoder.inverse_transform(np.round(predictions).astype(int).flatten())

X['prediction'] = predicted_labels

json_data = X.to_json(orient='records')

url = 'http://localhost:8000/submit'

response = requests.post(url, data={'key':json_data}, verify=False)

if response.status_code == 200:
    print(response.text)
else:
    print('POST request failed with status code:', response.status_code)

# save_model(model,"models/master_ann_model.h5")