
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor

try:
    import tensorflow as tf
except Exception: 
    tf = None
    Sequential = None
    LSTM = None
    Dense = None


def load_ebola_data(filename: str):
    """
    Load Ebola data where column 0 is a date string (YYYY-MM-DD)
    and column 2 is number of new cases.
    """
    data = np.genfromtxt(
        filename,
        delimiter=None,
        dtype=None,
        names=True,        # read header
        encoding="utf-8"
    )

    # Convert date strings into day indices (0, 1, 2, ...)
    date_strings = data[data.dtype.names[0]]  # first column name (e.g. 'Date')
    days = np.arange(len(date_strings))

    # new cases is usually col 2
    new_cases = data[data.dtype.names[2]]

    cumulative = np.cumsum(new_cases)

    return days, new_cases, cumulative



def plot_data_and_cumulative(days, new_cases, cumulative, country_name: str):
    """
    Plot new cases (scatter) and cumulative cases (line) for one country.
    """
    days = np.asarray(days)
    new_cases = np.asarray(new_cases)
    cumulative = np.asarray(cumulative)

    fig, ax1 = plt.subplots()

    ax1.scatter(days, new_cases, s=15, label="New cases")
    ax1.set_xlabel("Days since first outbreak")
    ax1.set_ylabel("Number of outbreaks")

    ax2 = ax1.twinx()
    ax2.plot(days, cumulative, label="Cumulative outbreaks")
    ax2.set_ylabel("Cumulative number of outbreaks")

    plt.title(f"Ebola outbreaks in {country_name}")
    fig.tight_layout()

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.show()



def fit_linear_regression(days , target):
    X = np.asarray(days).reshape(-1, 1)
    y = np.asarray(target)
    model = LinearRegression()
    model.fit(X, y)
    return model


def plot_linear_regression(days,  target, model, country_name, ylabel):
    X = np.asarray(days).reshape(-1, 1)
    y_pred = model.predict(X)

    plt.scatter(days, target, s=15, label="Data")
    plt.plot(days, y_pred, label="Linear fit")
    plt.xlabel("Days since first outbreak")
    plt.ylabel(ylabel)
    plt.title(f"Linear regression – {country_name}")
    plt.legend()
    plt.show()


def fit_polynomial_regression(days , target, degree=4):
    X = np.asarray(days ).reshape(-1, 1)
    y = np.asarray(target)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly


def plot_polynomial_regression(days,  target, model, poly, country_name, ylabel):
    days = np.asarray(days)
    X = days.reshape(-1, 1)
    X_poly = poly.transform(X)
    y_pred = model.predict(X_poly)

    plt.scatter(days, target, s=15, label="Data")
    plt.plot(days, y_pred, label=f"Polynomial fit (deg={poly.degree})")
    plt.xlabel("Days since first outbreak")
    plt.ylabel(ylabel)
    plt.title(f"Polynomial regression – {country_name}")
    plt.legend()
    plt.show()


def fit_mlp_regressor(days, target, hidden_layers=(20, 20), max_iter=2000):
    X = np.asarray(days).reshape(-1, 1)
    y = np.asarray(target)
    mlp = MLPRegressor(
        hidden_layer_sizes= hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=0,
    )
    mlp.fit(X, y)
    return mlp


def plot_mlp_regression(days,  target, mlp, country_name, ylabel):
    X = np.asarray(days).reshape(-1, 1)
    y_pred = mlp.predict(X)

    plt.scatter(days, target, s=15, label="Data")
    plt.plot(days,  y_pred, label="MLP prediction")
    plt.xlabel("Days since first outbreak")
    plt.ylabel(ylabel)
    plt.title(f"Neural network (MLP) – {country_name}")
    plt.legend()
    plt.show()


def build_sequences(series, window=10):
    data = np.asarray(series, dtype=float)
    X_list = []
    y_list = []
    for i in range(len(data) - window):
        X_list.append(data[i : i + window])
        y_list.append(data[i + window])
    X = np.array(X_list)
    y = np.array(y_list)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def fit_lstm(series, window=10, epochs=10, verbose=0):
    if Sequential is None or LSTM is None:
        raise RuntimeError("Tensorflow / Keras is not installed.")

    X, y = build_sequences(series, window=window)

    model = Sequential(
        [
            LSTM(32, activation="tanh", input_shape=(X.shape[1], 1)),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, verbose=verbose)
    y_pred = model.predict(X).flatten()
    return model, X, y, y_pred


def plot_lstm_results(y, y_pred, country_name):
    plt.plot(y, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.xlabel("Time step (sliding window index)")
    plt.ylabel("New cases")
    plt.title(f"LSTM predictions – {country_name}")
    plt.legend()
    plt.show()
