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
import polars as pl
from utils import PROJECT_ROOT
from absl import app


def get_tfidf_matrix(lazy_frame: LazyFrame) -> DataFrame:
    corpus = lazy_frame.collect()['question'].to_list()

    vectorizer = TfidfVectorizer(stop_words=['?'])
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1

    df = pl.DataFrame({
        "feature_names": feature_names,
        "score": scores
    })

    return df

def main(argv):
    data = get_dataset()
    dataframe = get_tfidf_matrix(data)

    dataframe.write_parquet(f"{PROJECT_ROOT}/data/tfidf_matrix.parquet")


if __name__ == '__main__':
  app.run(main)