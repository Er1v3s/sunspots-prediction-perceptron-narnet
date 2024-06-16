from pandas import read_csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

IMAGES_PATH = Path() / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id: object, tight_layout: object = True,
             fig_extension: object = "png", resolution: object = 300) -> object:
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

data_path = 'data/sunspot.csv'
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

netSize = [10, 20, 30]

for i in netSize:
    model_NARNET = Sequential([
        SimpleRNN(i, input_shape=(look_back, 1)),
        Dense(1)
    ])

    model_NARNET.compile(optimizer='adam', loss='mean_squared_error')
    model_NARNET.fit(X_train, Y_train, epochs=10, batch_size=1, verbose=0)

    y_pred_scaled = model_NARNET.predict(X_test)
    y_pred_NARNET = scaler.inverse_transform(y_pred_scaled)

    r2Perc = r2_score(scaler.inverse_transform(Y_test).flatten(), y_pred_NARNET.flatten())
    rmsePerc = np.sqrt(mean_squared_error(scaler.inverse_transform(Y_test).flatten(), y_pred_NARNET.flatten()))
    residuals_NARNET = scaler.inverse_transform(Y_test).flatten() - y_pred_NARNET.flatten()

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Time course
    axs[0].plot(years, sunspots, label='Liczba plam słonecznych')
    axs[0].set_title('Przebieg czasowy liczby plam słonecznych')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), shadow=True, ncol=5)

    # Narnet
    axs[1].plot(years[-len(y_pred_NARNET):], scaler.inverse_transform(Y_test),
                 label='Dane oryginalne')
    axs[1].plot(years[-len(y_pred_NARNET):], y_pred_NARNET,
                 label=f'Predicted (R^2={r2Perc:.2f}, RMSE={rmsePerc:.2f})', color='orange')
    axs[1].set_title('NARNET')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), shadow=True, ncol=5)

    # Residuals
    axs[2].hist(residuals_NARNET, bins=30, label='reszta', color='black', rwidth=0.5)
    axs[2].set_title('Histogram reszt')
    axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1), shadow=True, ncol=5)

    plt.tight_layout()
    plt.show()
    save_fig(f'NARNET_netsize={i}')


