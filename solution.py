import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
import numpy as np

# Adatok beolvasása
x_train = pd.read_csv('housing_x_train_167909.csv', sep=',', encoding='utf-8').values
y_train = pd.read_csv('housing_y_train_167909.csv', sep=',', encoding='utf-8').values
x_test = pd.read_csv('housing_x_test_167909.csv', sep=',', encoding='utf-8').values

# Adatok normalizálása StandardScaler segítségével
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_normalized = scaler_x.fit_transform(x_train)
y_train_normalized = scaler_y.fit_transform(y_train)

x_test_normalized = scaler_x.transform(x_test)

# Neurális hálózat létrehozása és tanítása
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))  # Dropout réteg a túltanulás elkerülésére
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))  # Dropout réteg a túltanulás elkerülésére
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train_normalized, y_train_normalized, epochs=150, batch_size=64, validation_split=0.2)

# Teszt adatokon predikció
predictions_normalized = model.predict(x_test_normalized)

# Predikciók visszaalakítása az eredeti skálázásra
predictions = scaler_y.inverse_transform(predictions_normalized)

# Predikciók .csv-be írása
np.savetxt('housing_y_test.csv', predictions, delimiter=",", fmt="%g")
