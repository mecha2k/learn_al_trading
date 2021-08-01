import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_financial_data(start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print("Reading GOOG data...")
    except FileNotFoundError:
        print("Downloading the GOOG data...")
        df = yf.download("GOOG", start=start_date, end=end_date)
        df.to_pickle(output_file)
    return df


if __name__ == "__main__":
    src_data = "../data/goog_data.pkl"
    goog_data = load_financial_data("2001-01-01", "2021-01-01", src_data)

    goog_data["Open-Close"] = goog_data.Open - goog_data.Close
    goog_data["High-Low"] = goog_data.High - goog_data.Low
    goog_data = goog_data.dropna()
    X = goog_data[["Open-Close", "High-Low"]]
    Y = np.where(goog_data["Close"].shift(-1) > goog_data["Close"], 1, -1)

    split_ratio = 0.8
    split_value = int(split_ratio * len(goog_data))
    X_train = X[:split_value]
    Y_train = Y[:split_value]
    X_test = X[split_value:]
    Y_test = Y[split_value:]

    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, Y_train)
    accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
    accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

    goog_data["Predicted_Signal"] = knn.predict(X)
    goog_data["GOOG_Returns"] = np.log(goog_data["Close"] / goog_data["Close"].shift(1))

    def calculate_return(df, split_value, symbol):
        cum_goog_return = df[split_value:]["%s_Returns" % symbol].cumsum() * 100
        df["Strategy_Returns"] = df["%s_Returns" % symbol] * df["Predicted_Signal"].shift(1)
        return cum_goog_return

    def calculate_strategy_return(df, split_value):
        cum_strategy_return = df[split_value:]["Strategy_Returns"].cumsum() * 100
        return cum_strategy_return

    cum_goog_return = calculate_return(goog_data, split_value=len(X_train), symbol="GOOG")
    cum_strategy_return = calculate_strategy_return(goog_data, split_value=len(X_train))

    def plot_chart(cum_symbol_return, cum_strategy_return, symbol):
        plt.figure(figsize=(10, 5))
        plt.plot(cum_symbol_return, label="%s Returns" % symbol)
        plt.plot(cum_strategy_return, label="Strategy Returns")
        plt.legend()
        plt.show()

    plot_chart(cum_goog_return, cum_strategy_return, symbol="GOOG")
    # print(accuracy_train, accuracy_test)
    #
    # goog_data["Predicted_Signal"] = knn.predict(X)
    # goog_data["GOOG_Returns"] = np.log(goog_data["Close"] / goog_data["Close"].shift(1))
    # cum_goog_return = goog_data[split_value:]["GOOG_Returns"].cumsum() * 100
    # goog_data["Strategy_Returns"] = goog_data["GOOG_Returns"] * goog_data["Predicted_Signal"].shift(
    #     1
    # )
    # cum_strategy_return = goog_data[split_value:]["Strategy_Returns"].cumsum() * 100
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(cum_goog_return, label="GOOG Returns")
    # plt.plot(cum_strategy_return, label="Strategy Returns")
    # plt.legend()
    # plt.show()

    def sharpe_ratio(symbol_returns, strategy_returns):
        strategy_std = strategy_returns.std()
        sharpe = (strategy_returns - symbol_returns) / strategy_std
        return sharpe.mean()

    print(sharpe_ratio(cum_strategy_return, cum_goog_return))
