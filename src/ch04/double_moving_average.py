import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_financial_data(start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print("Reading GOOG data...")
    except FileNotFoundError:
        print("Downloading the GOOG data...")
        df = yf.download("GOOG", start=start_date, end=end_date)
        df.to_pickle(output_file)
    return df


def double_moving_average(financial_data, short_window, long_window):
    signals = pd.DataFrame(index=financial_data.index)
    signals["signal"] = 0.0
    signals["short_mavg"] = (
        financial_data["Close"].rolling(window=short_window, min_periods=1, center=False).mean()
    )
    signals["long_mavg"] = (
        financial_data["Close"].rolling(window=long_window, min_periods=1, center=False).mean()
    )
    signals["signal"][short_window:] = np.where(
        signals["short_mavg"][short_window:] > signals["long_mavg"][short_window:], 1.0, 0.0
    )
    signals["orders"] = signals["signal"].diff()
    return signals


if __name__ == "__main__":
    src_data = "../data/data04-1.pkl"
    goog_data = load_financial_data("2001-01-01", "2021-01-01", src_data)

    ts = double_moving_average(goog_data, 20, 100)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel="Google price in $")
    goog_data["Adj Close"].plot(ax=ax1, color="g", lw=0.5)
    ts["short_mavg"].plot(ax=ax1, color="r", lw=2.0)
    ts["long_mavg"].plot(ax=ax1, color="b", lw=2.0)

    ax1.plot(
        ts.loc[ts.orders == 1.0].index,
        goog_data["Adj Close"][ts.orders == 1.0],
        marker="^",
        markersize=5,
        color="k",
    )
    ax1.plot(
        ts.loc[ts.orders == -1.0].index,
        goog_data["Adj Close"][ts.orders == -1.0],
        marker="v",
        markersize=5,
        color="k",
    )

    plt.legend(["Price", "Short mavg", "Long mavg", "Buy", "Sell"])
    plt.title("Double Moving Average Trading Strategy")
    plt.show()
