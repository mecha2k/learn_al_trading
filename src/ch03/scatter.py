import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def load_financial_data(start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print("File data found...reading GOOG data")
    except FileNotFoundError:
        print("File not found...downloading the GOOG data")
        df = yf.download("GOOG", start=start_date, end=end_date)
        df.to_pickle(output_file)
    return df


def create_regression_trading_condition(df):
    df["Open-Close"] = df.Open - df.Close
    df["High-Low"] = df.High - df.Low
    df["Target"] = df["Close"].shift(-1) - df["Close"]
    df = df.dropna()
    X = df[["Open-Close", "High-Low"]]
    Y = df[["Target"]]
    return df, X, Y


if __name__ == "__main__":
    src_data = "../data/goog_data.pkl"
    goog_data = load_financial_data("2001-01-01", "2021-01-01", src_data)

    create_regression_trading_condition(goog_data)
    pd.plotting.scatter_matrix(
        goog_data[["Open-Close", "High-Low", "Target"]], grid=True, diagonal="kde", alpha=0.5
    )
    plt.show()
