from pandas import read_csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

IMAGES_PATH = Path() / "../images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id: object, tight_layout: object = True,
             fig_extension: object = "png", resolution: object = 300) -> object:
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

data_path = '../data/sunspot.csv'
data = read_csv(data_path, header=None, sep=";")

if data.empty:
    print('Cannot find data file')
    exit(0)
else:
    print(data.head())

years = data[0].values
sunspots = data[1].values

scaler = MinMaxScaler(feature_range=(0, 1))
sunspots_scaled = scaler.fit_transform(sunspots.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(years, sunspots_scaled, test_size=0.2, shuffle=False)

x_train = np.array(x_train).reshape(-1, 1)
x_test = np.array(x_test).reshape(-1, 1)

def create_dataset(data, n):
    X, y = [], []
    for i in range(n, len(data)):
         X.append(data[i - n:i])
         y.append(data[i])
    return np.array(X), np.array(y)

look_back = 9
X_train, Y_train = create_dataset(y_train, look_back)
X_test, Y_test = create_dataset(y_test, look_back)

model_perceptron = Sequential([
    Dense(1, input_dim=look_back, activation='linear'),
    Dropout(0.15)
],)
model_perceptron.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
model_perceptron.fit(X_train, Y_train, epochs=10, verbose=0)

y_pred_scaled = model_perceptron.predict(X_test)
y_pred_perceptron = scaler.inverse_transform(y_pred_scaled)

r2Perc = r2_score(scaler.inverse_transform(Y_test).flatten(), y_pred_perceptron.flatten())
rmsePerc = np.sqrt(mean_squared_error(scaler.inverse_transform(Y_test).flatten(), y_pred_perceptron.flatten()))
residuals_perceptron = scaler.inverse_transform(Y_test).flatten() - y_pred_perceptron.flatten()

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Time course
axs[0].plot(years, sunspots, label='Number of sunspots')
axs[0].set_title('Time course of solar sunspots')
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), shadow=True, ncol=5)

# Perceptron
axs[1].plot(years[-len(y_pred_perceptron):], scaler.inverse_transform(Y_test), label='Original Data')
axs[1].plot(years[-len(y_pred_perceptron):], y_pred_perceptron, label=f'Predicted (R^2={r2Perc:.2f}, RMSE={rmsePerc:.2f})', color='red')
axs[1].set_title('Time course of solar sunspots')
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), shadow=True, ncol=5)

# Residuals
axs[2].hist(residuals_perceptron, bins=30, label="Residuals", color='black', rwidth=0.5)
axs[2].set_title('Histogram of residuals')
axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1), shadow=True, ncol=5)

plt.tight_layout()
plt.show()
save_fig("Perceptron")
