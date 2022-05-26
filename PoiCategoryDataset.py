import json
from tensorflow.keras import utils as np_utils
import spektral as sk

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spektral.data import Dataset, Graph


class PoiCategoryDataset(Dataset):
    """
    The PoICategory Dataset
    **Arguments**pip
    - `name`: str, name of the dataset to load.
    """

    def __init__(self, name, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def poi_dataset(self, max_samples=5000):

        A_df = pd.read_csv(
            "https://raw.githubusercontent.com/claudiocapanema/minicurso_gnn_sbrc2022/main/datasets/adjacency.zip",
            compression='zip').dropna(how='any', axis=0)
        X_df = pd.read_csv(
            "https://raw.githubusercontent.com/claudiocapanema/minicurso_gnn_sbrc2022/main/datasets/node_features.zip",
            compression='zip').dropna(how='any', axis=0)
        print("Original number of graphs", len(A_df))
        userid = A_df['user_id'].tolist()[:max_samples]
        matrix_df = A_df['matrices'].tolist()
        temporal_df = X_df['matrices'].tolist()
        category_df = A_df['category'].tolist()

        A_list = []
        X_list = []
        labels_labels_list = []
        count_nodes = 0

        for i in range(len(userid)):
            adjacency = matrix_df[i]
            labels = category_df[i]
            adjacency = json.loads(adjacency)
            if len(adjacency) < 2:
                continue

            labels = json.loads(labels)
            labels = np.array(labels)
            node_features = temporal_df[i]
            node_features = json.loads(node_features)
            node_features = np.array(node_features).astype(np.float)
            node_features = _normalize(node_features)
            adjacency = np.array(adjacency).astype(np.float)


            labels = np.array(np_utils.to_categorical(labels, num_classes=7))
            labels_labels_list.append(labels)

            """ Change the pre-processing based on the used message passing layer """
            adjacency = sk.layers.ARMAConv.preprocess(adjacency)
            count_nodes += len(adjacency)
            A_list.append(adjacency)
            X_list.append(node_features)

        print("Total of nodes: ", count_nodes)

        A_list, X_list, labels_list = np.array(A_list), np.array(X_list), np.array(labels_labels_list)

        print("A: ", A_list.shape, " X: ", X_list.shape, " Labels: ", labels_list.shape)

        return A_list, X_list, labels_list

    def read(self):

        # Convert to Graph
        a_list, x_list, labels = self.poi_dataset()
        print("Successfully loaded {}.".format(self.name))
        e_list = [None] * len(a_list)
        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]


def _normalize(x, norm=None):
    """
    Apply one-hot encoding or z-score to a list of node features
    """
    if norm == "ohe":
        fnorm = OneHotEncoder(sparse=False, categories="auto")
    elif norm == "zscore":
        fnorm = StandardScaler()
    else:
        return x
    return fnorm.fit_transform(x)