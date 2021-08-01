import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

"""
The Moving Average Convergence Divergence
 (MACD) was developed by Gerald Appel, and is based on the differences
 between two moving averages of different lengths, a Fast and a Slow moving
 average. A second line, called the Signal line is plotted as a moving
 average of the MACD. A third line, called the MACD Histogram is
 optionally plotted as a histogram of the difference between the
 MACD and the Signal Line.

 MACD = FastMA - SlowMA

Where:

FastMA is the shorter moving average and SlowMA is the longer moving average.
SignalLine = MovAvg (MACD)
MACD Histogram = MACD - SignalLine
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

    num_periods_fast = 10  # fast EMA time period
    K_fast = 2 / (num_periods_fast + 1)  # fast EMA smoothing factor
    ema_fast = 0
    num_periods_slow = 40  # slow EMA time period
    K_slow = 2 / (num_periods_slow + 1)  # slow EMA smoothing factor
    ema_slow = 0
    num_periods_macd = 20  # MACD EMA time period
    K_macd = 2 / (num_periods_macd + 1)  # MACD EMA smoothing factor
    ema_macd = 0

    ema_fast_values = []  # track fast EMA values for visualization purposes
    ema_slow_values = []  # track slow EMA values for visualization purposes
    macd_values = []  # track MACD values for visualization purposes
    macd_signal_values = []  # MACD EMA values tracker
    macd_historgram_values = []  # MACD - MACD-EMA
    for close_price in close:
        if ema_fast == 0:  # first observation
            ema_fast = close_price
            ema_slow = close_price
        else:
            ema_fast = (close_price - ema_fast) * K_fast + ema_fast
            ema_slow = (close_price - ema_slow) * K_slow + ema_slow

        ema_fast_values.append(ema_fast)
        ema_slow_values.append(ema_slow)

        macd = ema_fast - ema_slow  # MACD is fast_MA - slow_EMA
        if ema_macd == 0:
            ema_macd = macd
        else:
            ema_macd = (macd - ema_macd) * K_macd + ema_macd  # signal is EMA of MACD values

        macd_values.append(macd)
        macd_signal_values.append(ema_macd)
        macd_historgram_values.append(macd - ema_macd)

    goog_data = goog_data.assign(ClosePrice=pd.Series(close, index=goog_data.index))
    goog_data = goog_data.assign(
        FastExponential10DayMovingAverage=pd.Series(ema_fast_values, index=goog_data.index)
    )
    goog_data = goog_data.assign(
        SlowExponential40DayMovingAverage=pd.Series(ema_slow_values, index=goog_data.index)
    )
    goog_data = goog_data.assign(
        MovingAverageConvergenceDivergence=pd.Series(macd_values, index=goog_data.index)
    )
    goog_data = goog_data.assign(
        Exponential20DayMovingAverageOfMACD=pd.Series(macd_signal_values, index=goog_data.index)
    )
    goog_data = goog_data.assign(
        MACDHistorgram=pd.Series(macd_historgram_values, index=goog_data.index)
    )

    close_price = goog_data["ClosePrice"]
    ema_f = goog_data["FastExponential10DayMovingAverage"]
    ema_s = goog_data["SlowExponential40DayMovingAverage"]
    macd = goog_data["MovingAverageConvergenceDivergence"]
    ema_macd = goog_data["Exponential20DayMovingAverageOfMACD"]
    macd_histogram = goog_data["MACDHistorgram"]

    fig = plt.figure()
    plt.tight_layout()
    # plt.ticklabel_format()
    plt.rcParams.update({"font.size": 8})
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax1 = fig.add_subplot(311, ylabel="Google price in $")
    ax1.axes.xaxis.set_ticklabels([])
    close_price.plot(ax=ax1, color="g", lw=2.0, legend=True)
    ema_f.plot(ax=ax1, color="b", lw=2.0, legend=True)
    ema_s.plot(ax=ax1, color="r", lw=2.0, legend=True)
    ax2 = fig.add_subplot(312, ylabel="MACD")
    ax2.axes.xaxis.set_ticklabels([])
    macd.plot(ax=ax2, color="black", lw=2.0, legend=True)
    ema_macd.plot(ax=ax2, color="g", lw=2.0, legend=True)
    ax3 = fig.add_subplot(313, ylabel="MACD")
    ax3.axes.xaxis.set_ticklabels([])
    macd_histogram.plot(ax=ax3, color="r", kind="bar", legend=True, use_index=False)
    plt.show()
