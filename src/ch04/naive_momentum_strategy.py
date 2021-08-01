import yfinance as yf
import pandas as pd
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


def naive_momentum_trading(financial_data, nb_conseq_days):
    signals = pd.DataFrame(index=financial_data.index)
    signals["orders"] = 0
    cons_day = 0
    prior_price = 0
    init = True
    for k in range(len(financial_data["Adj Close"])):
        price = financial_data["Adj Close"][k]
        if init:
            prior_price = price
            init = False
        elif price > prior_price:
            if cons_day < 0:
                cons_day = 0
            cons_day += 1
        elif price < prior_price:
            if cons_day > 0:
                cons_day = 0
            cons_day -= 1
        if cons_day == nb_conseq_days:
            signals["orders"][k] = 1
        elif cons_day == -nb_conseq_days:
            signals["orders"][k] = -1
    return signals


if __name__ == "__main__":
    src_data = "../data/goog_data.pkl"
    goog_data = load_financial_data("2001-01-01", "2021-01-01", src_data)

    ts = naive_momentum_trading(goog_data, 5)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel="Google price in $")
    goog_data["Adj Close"].plot(ax=ax1, color="g", lw=0.5)

    ax1.plot(
        ts.loc[ts.orders == 1.0].index,
        goog_data["Adj Close"][ts.orders == 1],
        marker="^",
        markersize=7,
        color="k",
    )

    ax1.plot(
        ts.loc[ts.orders == -1.0].index,
        goog_data["Adj Close"][ts.orders == -1],
        marker="v",
        markersize=7,
        color="k",
    )

    plt.legend(["Price", "Buy", "Sell"])
    plt.title("Naive Momentum Trading Strategy")
    plt.show()
