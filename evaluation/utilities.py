import pandas as pd


def load_csv(df_file):
    return pd.read_csv(df_file, index_col=0)
