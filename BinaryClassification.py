#! /usr/bin/python3

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# File mit Testdaten
df = pd.read_csv('houseprices.csv')

# Anzahl der X- und Y-Felder aus der jeweiligen CSV

# Houseprices
xCount = 10

# # Iris
# xCount = 4

# Bei BinaryClassification ist die Anzahl der Outputs immer 1
yCount = 1

dataset = df.values

# Abhängig von der Anzahl der X-Felder die Arrays teilen (Index 0-13 = Input, Index 14 = Output)
x = dataset[:, 0:xCount]
Y = dataset[:, xCount]

# Scaler verwenden, um alle Werte in dem Bereich zwischen 0 und 1 zu skalieren, damit die Gewichtung nicht beeinflusst wird
min_max_scaler = preprocessing.MinMaxScaler()

X_scale = min_max_scaler.fit_transform(x)

# Daten in Trainingsdaten und Test- bzw. Validierungsdaten spliten
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
    X_scale, Y, test_size=0.3)

# Test- bzw. Validierungsdaten splitten in Testdaten und Validierungsdaten
X_val, X_test, Y_val, Y_test = train_test_split(
    X_val_and_test, Y_val_and_test, test_size=0.5)

model = Sequential([
    Dense(32, activation='relu', input_shape=(xCount,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

# Loss-Function = Binary_crossentropy setzen, um ein binäres Result zu erhalten
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model mit den Trainingsdaten trainieren
hist = model.fit(X_train, Y_train,
                 batch_size=32, epochs=100,
                 validation_data=(X_val, Y_val))

# Evaluierung des trainierten Models
result = model.evaluate(X_test, Y_test)

# Ausgabe in einem Plot (Jupyter-Notebook)
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper right')
# plt.show()

# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='lower right')
# plt.show()

# Prediction durchführen
result = model.predict(X_test)

sumCorrect = 0

# Berechnung der Anzahl richtiger Vorhersagen

for i in range(0, len(result)):
    yResult = Y_test[i]
    yRounded = round(result[i][0])
    yExact = result[i][0]
    if yResult == yRounded:
        sumCorrect = sumCorrect+1

calculatedAccuracy = round(100/len(result)*sumCorrect, 2)

print("Prozentualer Anteil an korrekten Predictions: ",
      calculatedAccuracy, " %")

# Die Genauigkeit der Predictions in ein File schreiben

outputFile = open("result.txt", "a")
outputFile.write(str(calculatedAccuracy)+"\n")
outputFile.close()
