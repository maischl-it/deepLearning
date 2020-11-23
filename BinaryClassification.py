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
xCount = 10
yCount = 1

dataset = df.values

x = dataset[:, 0:xCount]

Y = dataset[:, xCount]

min_max_scaler = preprocessing.MinMaxScaler()

X_scale = min_max_scaler.fit_transform(x)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
    X_scale, Y, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_val_and_test, Y_val_and_test, test_size=0.5)

model = Sequential([
    Dense(32, activation='relu', input_shape=(xCount,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
                 batch_size=32, epochs=100,
                 validation_data=(X_val, Y_val))

result = model.evaluate(X_test, Y_test)

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

result = model.predict(X_test)

sumCorrect = 0

for i in range(0, len(result)):
    yResult = Y_test[i]
    yRounded = round(result[i][0])
    yExact = result[i][0]
    if yResult == yRounded:
        sumCorrect = sumCorrect+1

calculatedAccuracy = round(100/len(result)*sumCorrect, 2)

print("Prozentualer Anteil an korrekten Predictions: ",
      calculatedAccuracy, " %")

outputFile = open("result.txt", "a")

outputFile.write(str(calculatedAccuracy)+"\n")

outputFile.close()
