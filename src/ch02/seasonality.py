import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA

if __name__ == "__main__":
    src_data = "../data/goog_data.pkl"
    try:
        google = pd.read_pickle(src_data)
    except FileNotFoundError:
        google = yf.download("GOOG", start="2001-01-01", end="2020-12-31")
        google.to_pickle(src_data)

    goog_data = google
    print(goog_data.info)

    goog_monthly_return = (
        goog_data["Adj Close"]
        .pct_change()
        .groupby([goog_data["Adj Close"].index.year, goog_data["Adj Close"].index.month])
        .mean()
    )

    goog_montly_return_list = []
    for i in range(len(goog_monthly_return)):
        goog_montly_return_list.append(
            {"month": goog_monthly_return.index[i][1], "monthly_return": goog_monthly_return[i]}
        )

    goog_montly_return_list = pd.DataFrame(
        goog_montly_return_list, columns=("month", "monthly_return")
    )

    goog_montly_return_list.boxplot(column="monthly_return", by="month")
    ax = plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticklabels(labels)
    ax.set_ylabel("GOOG return")
    plt.tick_params(axis="both", which="major", labelsize=7)
    plt.title("GOOG Montly return 2001-2018")
    plt.suptitle("")
    plt.show()

    fig = plt.figure()
    goog_data["Adj Close"].pct_change().groupby([goog_data["Adj Close"].index.month])
    ax1 = fig.add_subplot(111, ylabel="Monthly return")
    goog_monthly_return.plot()
    plt.xlabel("Time")
    plt.show()

    # Displaying rolling statistics
    def plot_rolling_statistics_ts(ts, titletext, ytext, window_size=12):
        ts.plot(color="red", label="Original", lw=0.5)
        ts.rolling(window_size).mean().plot(color="blue", label="Rolling Mean")
        ts.rolling(window_size).std().plot(color="black", label="Rolling Std")
        plt.legend(loc="best")
        plt.ylabel(ytext)
        plt.title(titletext)
        plt.show(block=False)

    plot_rolling_statistics_ts(
        goog_monthly_return[1:], "GOOG prices rolling mean and standard deviation", "Monthly return"
    )
    plot_rolling_statistics_ts(
        goog_data["Adj Close"],
        "GOOG prices rolling mean and standard deviation",
        "Daily prices",
        365,
    )

    plot_rolling_statistics_ts(
        goog_data["Adj Close"] - goog_data["Adj Close"].rolling(365).mean(),
        "GOOG prices without trend",
        "Daily prices",
        365,
    )

    def test_stationarity(timeseries):
        print("Results of Dickey-Fuller Test:")
        dftest = adfuller(timeseries[1:], autolag="AIC")
        dfoutput = pd.Series(
            dftest[0:4],
            index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"],
        )
        print(dfoutput)

    test_stationarity(goog_monthly_return[1:])
    test_stationarity(goog_data["Adj Close"])

    plt.figure()
    plt.subplot(211)
    plot_acf(goog_monthly_return[1:], ax=plt.gca(), lags=10)
    plt.subplot(212)
    plot_pacf(goog_monthly_return[1:], ax=plt.gca(), lags=10)
    plt.show()

    model = ARIMA(goog_monthly_return[1:], order=(2, 0, 2))
    fitted_results = model.fit()
    goog_monthly_return[1:].plot()
    fitted_results.fittedvalues.plot(color="red")
    plt.show()
