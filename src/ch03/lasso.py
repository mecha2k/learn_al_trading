import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model


def load_financial_data(start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print("Reading GOOG data...")
    except FileNotFoundError:
        print("Downloading the GOOG data...")
        df = yf.download("GOOG", start=start_date, end=end_date)
        df.to_pickle(output_file)
    return df


def create_classification_trading_condition(df):
    df["Open-Close"] = df.Open - df.Close
    df["High-Low"] = df.High - df.Low
    df = df.dropna()
    X = df[["Open-Close", "High-Low"]]
    Y = np.where(df["Close"].shift(-1) > df["Close"], 1, -1)
    return df, X, Y


def create_regression_trading_condition(df):
    df["Open-Close"] = df.Open - df.Close
    df["High-Low"] = df.High - df.Low
    df["Target"] = df["Close"].shift(-1) - df["Close"]
    df = df.dropna()
    X = df[["Open-Close", "High-Low"]]
    Y = df[["Target"]]
    return df, X, Y


def create_train_split_group(X, Y, split_ratio=0.8):
    return train_test_split(X, Y, shuffle=False, train_size=split_ratio)


if __name__ == "__main__":
    src_data = "../data/goog_data.pkl"
    goog_data = load_financial_data("2001-01-01", "2021-01-01", src_data)

    goog_data, X, Y = create_regression_trading_condition(goog_data)
    X_train, X_test, Y_train, Y_test = create_train_split_group(X, Y, split_ratio=0.8)

    lasso = linear_model.Lasso(alpha=0.1)
    lasso.fit(X_train, Y_train)
    print("Coefficients: \n", lasso.coef_)

    goog_data["Predicted_Signal"] = lasso.predict(X)
    goog_data["GOOG_Returns"] = np.log(goog_data["Close"] / goog_data["Close"].shift(1))
    print(goog_data.head())

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

    def sharpe_ratio(symbol_returns, strategy_returns):
        strategy_std = strategy_returns.std()
        sharpe = (strategy_returns - symbol_returns) / strategy_std
        return sharpe.mean()

    print(sharpe_ratio(cum_strategy_return, cum_goog_return))

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(Y_train, lasso.predict(X_train)))
    # Explained variance score: 1 is perfect prediction
    print("Variance score: %.2f" % r2_score(Y_train, lasso.predict(X_train)))

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(Y_test, lasso.predict(X_test)))
    # Explained variance score: 1 is perfect prediction
    print("Variance score: %.2f" % r2_score(Y_test, lasso.predict(X_test)))
