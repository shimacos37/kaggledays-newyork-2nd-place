import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import pandas as pd


def fix(df: pd.DataFrame) -> pd.DataFrame:
    broken = df["year"].isin(["DK Publishing Inc", "Gallimard"])
    author = df.loc[broken, "title"].str.split(";", expand=True)[1].str[:-1]
    year = df.loc[broken, "author"]
    publisher = df.loc[broken, "year"]

    df.loc[broken, "author"] = author
    df.loc[broken, "year"] = year
    df.loc[broken, "publisher"] = publisher

    df["author"] = df["author"].astype(str)
    df["year"] = df["year"].astype(int)
    df["publisher"] = df["publisher"].astype(str)
    return df


if __name__ == "__main__":
    os.makedirs("./input/preprocess", exist_ok=True)
    train_df = pd.read_csv("./input/train_ratings.csv")
    skf = KFold(n_splits=5, shuffle=True, random_state=777)
    for k, (_, valid_idx) in enumerate(skf.split(train_df)):
        train_df.loc[valid_idx, "fold"] = k
    test_df = pd.read_csv("./input/test_ratings.csv")
    book_df = pd.read_csv("./input/books.csv")
    book_df = fix(book_df)
    user_df = pd.read_csv("./input/users.csv")
    train_df = train_df.merge(book_df, on="book_id", how="left")
    test_df = test_df.merge(book_df, on="book_id", how="left")
    train_df = train_df.merge(user_df, on="user_id", how="left")
    test_df = test_df.merge(user_df, on="user_id", how="left")
    train_df.to_csv("./input/preprocess/train_ratings.csv", index=False)
    test_df.to_csv("./input/preprocess/test_ratings.csv", index=False)
