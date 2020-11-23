from keras import models, layers
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from sklearn.model_selection import train_test_split

# File laden (Trennzeichen sind mehrere Leerzeichen (Regex = \s+)
df = read_csv('housing.csv', delimiter=r"\s+")

# Anzahl der X- und Y-Felder aus der jeweiligen CSV
xCount = 13
yCount = 1

dataset = df.values

# Abhängig von der Anzahl der X-Felder die Arrays teilen (Index 0-13 = Input, Index 14 = Output)
x = dataset[:, 0:xCount]
y = dataset[:, xCount]

# Daten in Trainingsdaten und Test- bzw. Validierungsdaten spliten
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Scaler verwenden, damit die Gewichtung nicht beeinflusst wird
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = models.Sequential()

model.add(layers.Dense(8, activation='relu', input_shape=(xCount,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

# Loss-Function = mse setzen, um ein Regression-Result zu erhalten
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Model mit den Trainingsdaten trainieren
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=200)

# Evaluierung des trainierten Models
model.evaluate(X_test_scaled, y_test)

# Daten für die Prediction aus dem Array selektieren (Items von Index 0 bis 5)
to_predict = X_train_scaled[:5]

# Prediction durchführen
predictions = model.predict(to_predict)

# Ausgabe der Prediction und des tatsächlichen Ergebnisses
print(predictions)
print(y_train[:5])
