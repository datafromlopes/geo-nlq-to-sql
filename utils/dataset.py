# coding=utf-8
# Copyright (C) 2025  Diego Lopes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#     https://www.gnu.org/licenses/gpl-3.0.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from pathlib import Path
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

BASE_DATASET_NAME = config["dataset"]["base_name"]
DATASET_NAME = config["dataset"]["name"]
DATASET_TYPE = config["dataset"]["type"]
DATASET_PATH = PROJECT_ROOT / config["dataset"]["path"]
BASE_DATASET_FULL_NAME = f"{DATASET_PATH}/{BASE_DATASET_NAME}.{DATASET_TYPE}"
DATASET_FULL_NAME = f"{DATASET_PATH}/{DATASET_NAME}.{DATASET_TYPE}"


def get_dataset(base_dataset=False):
    """Get the dataset.

    Arguments:
        base_dataset (bool, optional): if True, returns the base dataset. Default is False.

    Returns:
        pl.LazyFrame: Porlars LazyFrame
    """

    if base_dataset:
        return pl.scan_parquet(BASE_DATASET_FULL_NAME)
    return pl.scan_parquet(DATASET_FULL_NAME)