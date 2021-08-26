import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import coint


def load_financial_data(symbols, start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print("Reading symbols data...")
    except FileNotFoundError:
        print("Downloading the symbols data...")
        df = yf.download(symbols, start=start_date, end=end_date)
        df.to_pickle(output_file)
    return df


def find_cointegrated_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            result = coint(data[keys[i]], data[keys[j]])
            pvalue_matrix[i, j] = result[1]
            if result[1] < 0.02:
                pairs.append((keys[i], keys[j]))
    return pvalue_matrix, pairs


if __name__ == "__main__":
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    src_data = "../data/multi_data_large.pkl"
    symbolsIds = ["SPY", "AAPL", "ADBE", "LUV", "MSFT", "SKYW", "QCOM", "HPQ", "JNPR", "AMD", "IBM"]
    data = load_financial_data(symbolsIds, "2001-01-01", "2021-01-01", src_data)

    pvalues, pairs = find_cointegrated_pairs(data["Adj Close"])
    print(pairs)
    print(data.head(3))

    sns.heatmap(
        pvalues,
        xticklabels=symbolsIds,
        yticklabels=symbolsIds,
        cmap="RdYlGn_r",
        mask=(pvalues >= 0.98),
    )
    plt.show()
