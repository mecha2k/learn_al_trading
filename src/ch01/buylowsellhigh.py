import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    start = "2014-01-01"
    end = "2015-01-01"
    google = yf.download("GOOG", start=start, end=end)
    print(google.info)

    google_signal = pd.DataFrame(index=google.index)
    google_signal["price"] = google["Adj Close"]
    google_signal["daily_difference"] = google_signal["price"].diff()
    google_signal["signal"] = google_signal["daily_difference"].apply(
        lambda x: 1.0 if x > 0 else 0.0
    )
    # google_signal["signal"] = 0.0
    # google_signal["signal"][:] = np.where(google_signal["daily_difference"][:] > 0, 1.0, 0.0)
    google_signal["positions"] = google_signal["signal"].diff()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel="Google price in $")
    google_signal["price"].plot(ax=ax1, color="r", lw=2.0)
    ax1.plot(
        google_signal.loc[google_signal.positions == 1.0].index,
        google_signal.price[google_signal.positions == 1.0],
        "^",
        markersize=3,
        color="m",
    )
    ax1.plot(
        google_signal.loc[google_signal.positions == -1.0].index,
        google_signal.price[google_signal.positions == -1.0],
        "v",
        markersize=3,
        color="g",
    )
    plt.show()

    # Set the initial capital
    initial_capital = float(1000.0)
    positions = pd.DataFrame(index=google_signal.index).fillna(0.0)
    portfolio = pd.DataFrame(index=google_signal.index).fillna(0.0)

    positions["GOOG"] = google_signal["signal"]
    portfolio["positions"] = positions.multiply(google_signal["price"], axis=0)
    portfolio["cash"] = (
        initial_capital - (positions.diff().multiply(google_signal["price"], axis=0)).cumsum()
    )
    portfolio["total"] = portfolio["positions"] + portfolio["cash"]
    portfolio.plot()
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel="Portfolio value in $")
    portfolio["total"].plot(ax=ax1, lw=2.0)
    ax1.plot(
        portfolio.loc[google_signal.positions == 1.0].index,
        portfolio.total[google_signal.positions == 1.0],
        "^",
        markersize=3,
        color="m",
    )
    ax1.plot(
        portfolio.loc[google_signal.positions == -1.0].index,
        portfolio.total[google_signal.positions == -1.0],
        "v",
        markersize=3,
        color="k",
    )
    plt.show()
