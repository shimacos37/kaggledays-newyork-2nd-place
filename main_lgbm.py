import logging
import os
import random
import pickle
from glob import glob
from itertools import combinations
from typing import List, Optional, Union, Dict, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from google.cloud import storage
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from src.factories.tree_factory import LGBMModel


plt.style.use("ggplot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepair_dir(config: DictConfig) -> None:
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
    ]:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(config_path="yamls", config_name="tree.yaml")
def main(config: DictConfig) -> None:
    os.chdir(config.workdir)
    set_seed(config.seed)
    train_df = pd.read_pickle(config.data.train_path)
    test_df = pd.read_pickle(config.data.test_path)
    prepair_dir(config)
    label_col = config.feature.label_col
    pred_col = config.feature.pred_col
    model = LGBMModel(config.lgbm)
    train_df[label_col] = train_df[label_col] - 1
    train_df, test_df = model.cv(train_df, test_df)

    model.save_model(config.store.model_path)
    model.save_importance(config.store.result_path)
    if config.lgbm.params.objective == "multiclass":
        prob_cols = [
            f"{label_col}_prob{c}" for c in range(config.lgbm.params.num_class)
        ]
        for c in range(config.lgbm.params.num_class):
            test_df[f"{label_col}_prob{c}"] = test_df[
                [f"{label_col}_prob{c}_fold{i}" for i in range(config.lgbm.n_fold)]
            ].mean(1)
        test_df[label_col] = (test_df[prob_cols].to_numpy().argmax(1)) + 1
        train_df[pred_col] = train_df[prob_cols].to_numpy().argmax(1)
        score = f1_score(train_df[pred_col], train_df[label_col], average="micro")
    else:
        train_df[f"{pred_col}_bin"] = np.clip(
            np.round(train_df[pred_col].values), 1, 10
        )
        score = f1_score(
            train_df[f"{pred_col}_bin"], train_df[label_col], average="micro"
        )
        test_df[label_col] = test_df[
            [f"{pred_col}_fold{i}" for i in range(config.lgbm.n_fold)]
        ].mean(1)
        score = f1_score(
            train_df[f"{pred_col}_bin"], train_df[label_col], average="micro"
        )
    train_df[label_col] = train_df[pred_col] + 1
    train_df[["id", label_col] + prob_cols].to_csv(
        f"{config.store.save_path}/valid.csv", index=False
    )
    test_df[["id", label_col] + prob_cols].to_csv(
        f"{config.store.save_path}/test.csv", index=False
    )
    test_df[["id", label_col]].to_csv(f"{config.store.save_path}/sub.csv", index=False)
    logger.info(f"fold all, score: {score}")


if __name__ == "__main__":
    main()
