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
import polars as pl
from scipy.sparse import load_npz, csr_matrix
from .global_variables import DATASET_FULL_NAME, TF_IDF_MATRIX_FULL_NAME


def get_dataset(base_dataset=False) -> pl.LazyFrame:
    """Get the dataset.

    Arguments:
        base_dataset (bool, optional): if True, returns the base dataset. Default is False.

    Returns:
        pl.LazyFrame: Porlars LazyFrame
    """
    return pl.scan_parquet(DATASET_FULL_NAME)

def get_tfidf_matrix() -> csr_matrix:
    """Get the TF-IDF matrix.

        Returns:
            scipy.csr_matrix: Sparse Matrix
        """

    return load_npz(TF_IDF_MATRIX_FULL_NAME)

