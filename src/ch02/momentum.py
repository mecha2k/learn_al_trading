import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

"""
The Momentum (MOM) indicator compares the
 current price with the previous price from a selected number of
 periods ago. This indicator is similar to the “Rate of Change” indicator,
 but the MOM does not normalize the price, so different instruments
 can have different indicator values based on their point values.

 MOM =  Price - Price of n periods ago
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

    time_period = 20  # how far to look back to find reference price to compute momentum
    history = []  # history of observed prices to use in momentum calculation
    mom_values = []  # track momentum values for visualization purposes
    for close_price in close:
        history.append(close_price)
        if len(history) > time_period:  # history is at most 'time_period' number of observations
            del history[0]
        mom = close_price - history[0]
        mom_values.append(mom)

    goog_data = goog_data.assign(ClosePrice=pd.Series(close, index=goog_data.index))
    goog_data = goog_data.assign(
        MomentumFromPrice20DaysAgo=pd.Series(mom_values, index=goog_data.index)
    )
    close_price = goog_data["ClosePrice"]
    momentum = goog_data["MomentumFromPrice20DaysAgo"]

    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel="Google price in $")
    close_price.plot(ax=ax1, color="g", lw=2.0, legend=True)
    ax2 = fig.add_subplot(212, ylabel="Momentum in $")
    momentum.plot(ax=ax2, color="b", lw=2.0, legend=True)
    plt.show()
