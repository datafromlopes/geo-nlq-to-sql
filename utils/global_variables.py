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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

dataset_name = config["dataset"]["name"]
dataset_type = config["dataset"]["type"]
dataset_path = PROJECT_ROOT / config["dataset"]["path"]
tfidf_matrix_name = config["dataset"]["tf_idf_matrix_name"]
tf_idf_matrix_type = config["dataset"]["tf_idf_matrix_type"]

DATASET_FULL_NAME = f"{dataset_path}/{dataset_name}.{dataset_type}"
TF_IDF_MATRIX_FULL_NAME = f"{dataset_path}/{tfidf_matrix_name}.{tf_idf_matrix_type}"