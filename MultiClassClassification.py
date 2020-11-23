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

xCount = 4
yCount = 1

dataset = df.values

# Abhängig von der Anzahl der X-Felder die Arrays teilen (Index 0-4 = Input, Index 5= Output)
x = dataset[:, 0:xCount]
Y = dataset[:, xCount]

# Die möglichen Ergebnisse (z. B. setosa als Index in einem Array darstellen z. B. setosa = [1,0,0] oder versicolor = [0,1,0])
encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(Y.reshape(-1, 1))

# Scaler verwenden, um alle Werte in dem Bereich zwischen 0 und 1 zu skalieren, damit die Gewichtung nicht beeinflusst wird
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(x)

# Daten in Trainingsdaten und Test- bzw. Validierungsdaten spliten
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
    X_scale, Y, test_size=0.3)

# Test- bzw. Validierungsdaten splitten in Testdaten und Validierungsdaten
X_val, X_test, Y_val, Y_test = train_test_split(
    X_val_and_test, Y_val_and_test, test_size=0.5)

# Für MultiClass-Classifications wird der Softmax-Activator verwendet
model = Sequential([
    Dense(10, activation='relu', input_shape=(xCount,)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax'),
])

# Loss-Function = categorical_crossentropy setzen, um ein MultiClass-Result zu erhalten
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model mit den Trainingsdaten trainieren
hist = model.fit(X_train,
                 Y_train,
                 batch_size=32,
                 epochs=100,
                 verbose=3,
                 validation_data=(X_val, Y_val))

# Evaluierung des trainierten Models
model.evaluate(X_test, Y_test)

# Prediction durchführen
result = model.predict(X_test)

sumCorrect = 0

# Berechnung der Anzahl richtiger Vorhersagen

for i in range(0, len(result)):
    yResult = Y_test[i]

    yRounded = [round(result[i][0]), round(result[i][1]), round(result[i][2])]

    if numpy.array_equal(yResult, yRounded):
        sumCorrect = sumCorrect+1


calculatedAccuracy = round(100/len(result)*sumCorrect, 2)

print("Prozentualer Anteil an korrekten Predictions: ",
      calculatedAccuracy, " %")

# Die Genauigkeit der Predictions in ein File schreiben
outputFile = open("result.txt", "a")
outputFile.write(str(calculatedAccuracy)+"\n")
outputFile.close()
