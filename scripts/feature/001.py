import datetime
import os
import re
import time
import yaml
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm


@contextmanager
def timer(name):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def fill_nan(df, col):
    df[col] = df[col].fillna("None")
    return df


def target_encoding(df, test_df, groupby_cols, target_col="target", n_fold=5):
    dfs = []
    for n_fold in range(n_fold):
        train_df = df.query("fold != @n_fold")
        valid_df = df.query("fold == @n_fold")
        feature = train_df.groupby(groupby_cols)[target_col].agg(
            ["median", "mean", "std", "min", "max"]
        )
        feature.columns = [
            "_".join(groupby_cols) + f"_{target_col}_" + col for col in feature.columns
        ]
        valid_df = valid_df.merge(feature, on=groupby_cols, how="left")
        dfs.append(valid_df)
    dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
    feature = df.groupby(groupby_cols)[target_col].agg(
        ["median", "mean", "std", "min", "max"]
    )
    feature.columns = [
        "_".join(groupby_cols) + f"_{target_col}_" + col for col in feature.columns
    ]
    test_df = test_df.merge(feature, on=groupby_cols, how="left")
    return dfs, test_df


def count_encoding(train_df, test_df, all_df):
    use_columns = [
        "title",
        "year",
        "city",
        "publisher",
        "country",
        "province",
    ]
    for col in use_columns:
        feature = all_df.groupby(col)[col].count()
        feature.name = f"{col}_count"
        train_df = train_df.merge(feature, on=col, how="left")
        test_df = test_df.merge(feature, on=col, how="left")
    return train_df, test_df


def agg_by_all_df(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_df: pd.DataFrame,
    groupby_cols: List[str],
    agg_cols: List[str],
    agg_funs: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature = all_df.groupby(groupby_cols)[agg_cols].agg(agg_funs)
    feature.columns = ["_".join(cols) for cols in feature.columns]
    feature = feature.add_suffix(f"_by_{'_'.join(groupby_cols)}")
    df = df.merge(feature, on=groupby_cols, how="left")
    test_df = test_df.merge(feature, on=groupby_cols, how="left")
    return df, test_df


def aggregate_feature(
    df: pd.DataFrame,
    groupby_cols: List[str],
    agg_cols: List[str],
    agg_funs: List[str],
) -> pd.DataFrame:
    feature = df.groupby(groupby_cols)[agg_cols].agg(agg_funs)
    feature.columns = ["_".join(cols) for cols in feature.columns]
    feature = feature.add_suffix(f"_by_{'_'.join(groupby_cols)}")
    df = df.merge(feature, on=groupby_cols, how="left")
    return df


def main():
    VERSION = re.split("[._]", os.path.basename(__file__))[-2]
    os.makedirs(f"./input/{VERSION}", exist_ok=True)
    cat_cols = [
        "title",
        "author",
        "publisher",
        "age_bin",
        "city",
        "province",
        "country",
        "user_id",
        "book_id",
    ]
    with timer("Prepare Data"):
        df = pd.read_csv("./input/preprocess/train_ratings.csv")
        test_df = pd.read_csv("./input/preprocess/test_ratings.csv")
        all_df = pd.concat([df, test_df], axis=0).reset_index(drop=True)
        df["age_bin"] = pd.cut(df["age"], np.arange(0, 100, 10)).cat.codes
        test_df["age_bin"] = pd.cut(test_df["age"], np.arange(0, 100, 10)).cat.codes

    with timer("count_encoding"):
        df, test_df = count_encoding(df, test_df, all_df)

    with timer("Label Encoding"):
        for col in tqdm(cat_cols):
            le = LabelEncoder()
            overlap_cat = set(df[col].unique()) & set(test_df[col].unique())

            # None埋め
            df = fill_nan(df, col)
            test_df = fill_nan(test_df, col)

            # 被ってるカテゴリ以外その他とする
            df.loc[df.query(f"{col} not in @overlap_cat").index, col] = "other"
            test_df.loc[
                test_df.query(f"{col} not in @overlap_cat").index, col
            ] = "other"

            # string型に変換
            df[col] = df[col].astype("str")
            test_df[col] = test_df[col].astype("str")

            df[col] = le.fit_transform(df[col])
            test_df[col] = le.transform(test_df[col])

    with timer("Target Encoding"):
        target_encode_cols = [
            "user_id",
            "book_id",
            "age_bin",
            "year",
            "country",
        ]
        for col in tqdm(target_encode_cols):
            df, test_df = target_encoding(df, test_df, [col], target_col="rating")
        for col_a, col_b in tqdm(combinations(cat_cols, 2)):
            df, test_df = target_encoding(
                df, test_df, [col_a, col_b], target_col="rating"
            )

    with timer("DIFF Feature"):
        mean_cols = [col for col in df.columns if col.endswith("mean")]
        for col_a, col_b in tqdm(combinations(mean_cols, 2)):
            df[f"diff_{col_a}_{col_b}"] = df[col_a] - df[col_b]
            test_df[f"diff_{col_a}_{col_b}"] = test_df[col_a] - test_df[col_b]

    feature_cols = [col for col in test_df.columns if col not in ["id"]]
    df.to_pickle(
        f"./input/{VERSION}/train.pkl.gz",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    test_df.to_pickle(
        f"./input/{VERSION}/test.pkl.gz",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    feature_dict = {
        "version": VERSION,
        "label_col": "rating",
        "pred_col": "rating_pred",
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
    }
    with open(f"./yamls/feature/{VERSION}.yaml", "w") as f:
        yaml.dump(feature_dict, f)
    print("feature_cols:", feature_cols)
    print("cat_cols:", cat_cols)


if __name__ == "__main__":
    main()
