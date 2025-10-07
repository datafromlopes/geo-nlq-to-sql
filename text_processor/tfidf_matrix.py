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
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import get_dataset
from polars import LazyFrame, DataFrame
from scipy.sparse import save_npz
from utils import PROJECT_ROOT
from absl import app


def get_tfidf_matrix(lazy_frame: LazyFrame) -> tuple:
    corpus = lazy_frame.collect()['question'].to_list()

    vectorizer = TfidfVectorizer()
    sparse_matrix = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()

    return sparse_matrix, features

def main(argv):
    docs = get_dataset()
    sparse_matrix, features = get_tfidf_matrix(docs)
    df_features = DataFrame({'features': features})

    df_features.write_parquet(f"{PROJECT_ROOT}/data/features.parquet")
    save_npz(f"{PROJECT_ROOT}/data/sparse_matrix.npz", sparse_matrix)


if __name__ == '__main__':
  app.run(main)