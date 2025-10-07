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
from pyspark.sql import SparkSession, DataFrame
from scipy.sparse import load_npz, csr_matrix, save_npz
from .global_variables import (
    DATASET_FULL_NAME,
    BASE_DATASET_FULL_NAME,
    TF_IDF_MATRIX_NAME,
    TF_IDF_FEATURES_NAME
)


class Dataset:
    def __init__(self):
        self.__spark = SparkSession.builder.appName("Dataset").getOrCreate()

    def get_dataset(self, base_dataset=False, partition=None) -> DataFrame:
        """Get the dataset.

        Arguments:
            base_dataset (bool, optional): if True, returns the base dataset. Default is False.
            partition (str, optional): if not None, returns the partition. Default is None.

        Returns:
            DataFrame: PySpark DataFrame
        """
        if base_dataset:
            return self.__spark.read.parquet(BASE_DATASET_FULL_NAME)

        if partition:
            file = f"{DATASET_FULL_NAME}/source={partition}"
            return self.__spark.read.parquet(file)

        return self.__spark.read.parquet(DATASET_FULL_NAME)


class TfIdfSparseMatrix:
    def __init__(self):
        pass

    @staticmethod
    def get_tfidf_matrix() -> csr_matrix:
        """Get the TF-IDF matrix.

        Returns:
            scipy.csr_matrix: Sparse Matrix
        """

        return load_npz(TF_IDF_MATRIX_NAME)

    @staticmethod
    def save_tfidf_matrix(sparse_matrix: csr_matrix, features: DataFrame) -> None:
        """Save the TF-IDF matrix and features."""
        save_npz(TF_IDF_MATRIX_NAME, sparse_matrix)
        features.write.mode("overwrite").parquet(TF_IDF_FEATURES_NAME)