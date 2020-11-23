#! /usr/bin/python3

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import numpy

# File mit Testdaten
df = pd.read_csv('iris.csv')

# Anzahl der X- und Y-Felder aus der jeweiligen CSV

# # Iris
xCount = 4

# Bei BinaryClassification ist die Anzahl der Outputs immer 1
yCount = 1

dataset = df.values

x = dataset[:, 0:xCount]

Y = dataset[:, xCount]

encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(Y.reshape(-1, 1))

min_max_scaler = preprocessing.MinMaxScaler()

X_scale = min_max_scaler.fit_transform(x)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
    X_scale, Y, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_val_and_test, Y_val_and_test, test_size=0.5)

model = Sequential([
    Dense(10, activation='relu', input_shape=(xCount,)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train,
                 Y_train,
                 batch_size=32,
                 epochs=100,
                 verbose=3,
                 validation_data=(X_val, Y_val))

model.evaluate(X_test, Y_test)

result = model.predict(X_test)

sumCorrect = 0

for i in range(0, len(result)):
    yResult = Y_test[i]

    yRounded = [round(result[i][0]), round(result[i][1]), round(result[i][2])]

    if numpy.array_equal(yResult, yRounded):
        sumCorrect = sumCorrect+1


calculatedAccuracy = round(100/len(result)*sumCorrect, 2)

print("Prozentualer Anteil an korrekten Predictions: ",
      calculatedAccuracy, " %")

outputFile = open("result.txt", "a")

outputFile.write(str(calculatedAccuracy)+"\n")

outputFile.close()
