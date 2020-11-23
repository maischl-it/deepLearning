from keras import models, layers
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from sklearn.model_selection import train_test_split

df = read_csv('housing.csv', delimiter=r"\s+")

xCount = 13

yCount = 1

dataset = df.values

x = dataset[:, 0:xCount]

y = dataset[:, xCount]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3)


scaler = StandardScaler()

# first we fit the scaler on the training dataset
scaler.fit(X_train)

# then we call the transform method to scale both the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# a sample output
print(X_train_scaled[0])


model = models.Sequential()

model.add(layers.Dense(8, activation='relu', input_shape=[X_train.shape[1]]))
model.add(layers.Dense(16, activation='relu'))

# output layer
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100)

model.evaluate(X_test_scaled, y_test)

to_predict = X_train_scaled[:5]

predictions = model.predict(to_predict)

print(predictions)

print(y_train[:5])
