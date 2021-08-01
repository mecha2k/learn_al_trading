import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats


"""
The Simple Moving Average (SMA) is calculated
 by adding the price of an instrument over a number of time periods
 and then dividing the sum by the number of time periods. The SMA
 is basically the average price of the given time period, with equal
 weighting given to the price of each period.

Simple Moving Average
SMA = ( Sum ( Price, n ) ) / n    

Where: n = Time Period
"""

if __name__ == "__main__":
    src_data = "../data/goog_data.pkl"
    try:
        google = pd.read_pickle(src_data)
    except FileNotFoundError:
        google = yf.download("GOOG", start="2014-01-01", end="2020-12-31")
        google.to_pickle(src_data)

    print(google.info)
    goog_data = google.tail(620)
    close = goog_data["Close"]

    time_period = 20  # number of days over which to average
    history = []  # to track a history of prices
    sma_values = []  # to track simple moving average values
    for close_price in close:
        history.append(close_price)
        if (
            len(history) > time_period
        ):  # we remove oldest price because we only average over last 'time_period' prices
            del history[0]
        sma_values.append(stats.mean(history))

    goog_data = goog_data.assign(ClosePrice=pd.Series(close, index=goog_data.index))
    goog_data = goog_data.assign(
        Simple20DayMovingAverage=pd.Series(sma_values, index=goog_data.index)
    )
    close_price = goog_data["ClosePrice"]
    sma = goog_data["Simple20DayMovingAverage"]

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel="Google price in $")
    close_price.plot(ax=ax1, color="g", lw=2.0, legend=True)
    sma.plot(ax=ax1, color="r", lw=2.0, legend=True)
    plt.show()
