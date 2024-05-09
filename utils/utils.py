from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial import distance
from scipy.special import kl_div
from scipy.stats import pearsonr
import numpy as np
import torch


class MakeTorchData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = float(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Metrics:

    @staticmethod
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        cosine_sim = cosine_similarity(preds, labels)
        euclidean_dist = euclidean_distances(preds, labels)
        pearson_corr, _ = pearsonr(preds.flatten(), labels.flatten())
        kl_divergence = np.array([kl_div(pred, label) for pred, label in zip(preds, labels)])

        def jaccard_similarity(x, y):
            intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
            union_cardinality = len(set.union(*[set(x), set(y)]))
            return intersection_cardinality / float(union_cardinality)

        jaccard_sim = np.array([jaccard_similarity(pred, label) for pred, label in zip(preds, labels)])

        return {
            "cosine_similarity": np.mean(cosine_sim),
            "euclidean_distance": np.mean(euclidean_dist),
            "jaccard_similarity": np.mean(jaccard_sim),
            "pearson_correlation": pearson_corr,
            "kl_divergence": np.mean(kl_divergence)
        }
