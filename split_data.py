import pandas as pd
from sklearn.model_selection import train_test_split

raw_df = pd.read_csv("data/World Happiness Report 2005-2021.csv")
raw_df.dropna(inplace=True)

train_df, test_df = train_test_split(raw_df, test_size=0.2, shuffle=True)

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)