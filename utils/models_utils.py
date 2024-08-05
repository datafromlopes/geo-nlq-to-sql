from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.special import kl_div
from scipy.stats import pearsonr
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }


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
            "kl_divergence": np.mean(kl_divergence.flatten())
        }

