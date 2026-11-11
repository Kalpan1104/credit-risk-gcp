import pandas as pd
import pytest

from credit_risk import config as C
from credit_risk.data import generate, write_splits
from credit_risk.model import train


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return write_splits(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def train_df(data_dir):
    return pd.read_csv(data_dir / "train.csv")


@pytest.fixture(scope="session")
def holdout_df(data_dir):
    return pd.read_csv(data_dir / "holdout.csv")


@pytest.fixture(scope="session")
def bundle(train_df):
    return train(train_df)


@pytest.fixture
def applicants():
    return generate(5, seed=9).drop(columns=[C.TARGET]).to_dict("records")
