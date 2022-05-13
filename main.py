import numpy as np
import pandas as pd
import sklearn.metrics as skm
from PoiCategoryDataset import PoiCategoryDataset

from spektral.data import BatchLoader, PackedBatchLoader

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import ARMAConv, GraphMasking, GCNConv
from spektral.transforms import LayerPreprocess

class GNN(Model):
    def __init__(self):
        super().__init__()
        self.mask = GraphMasking()
        self.conv1 = ARMAConv(
        16,
        iterations=1,
        order=2,
        share_weights=True,
        dropout_rate=0.75,
        activation="elu",
        gcn_activation="elu",
        kernel_regularizer=l2(5e-5)
    )
        self.dropout = Dropout(0.6)
        self.conv2 = ARMAConv(
        7,
        iterations=1,
        order=1,
        share_weights=True,
        dropout_rate=0.75,
        activation="softmax",
        gcn_activation=None,
        kernel_regularizer=l2(5e-5),
    )
    def call(self, inputs):
        X_input, A_input = inputs
        X = self.mask(X_input)
        X = self.conv1([X, A_input])
        X = self.dropout(X)
        output = self.conv2([X, A_input])
        return output

if __name__ == '__main__':

    dataset = PoiCategoryDataset("PoICategoryDataset", n_samples=10000)

    # Parameters
    N = max(g.n_nodes for g in dataset)
    D = dataset.n_node_features  # Dimension of node features
    S = dataset.n_edge_features  # Dimension of edge features
    n_out = dataset.n_labels  # Dimension of the target

    print("Parameters")
    print(N, D, S, n_out)

    np.random.seed(seed=1)
    # shuffle data
    idxs = np.random.permutation(len(dataset))
    split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
    idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
    dataset_train = dataset[idx_tr]
    dataset_validation = dataset[idx_va]
    dataset_test = dataset[idx_te]
    batch_size = 5  # Batch size
    epochs = 10
    # The data have already been shuffled
    loader_tr = BatchLoader(dataset_train, epochs=10, batch_size=batch_size, mask=True, node_level=True,
                            shuffle=False)
    loader_va = BatchLoader(dataset_validation, epochs=10, batch_size=batch_size, mask=True, node_level=True,
                            shuffle=False)
    loader_te = BatchLoader(dataset_test, epochs=10, batch_size=batch_size, mask=True, node_level=True,
                            shuffle=False)

    model = GNN()
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["acc"])

    model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=epochs,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        callbacks=[EarlyStopping(patience=3,
                                 restore_best_weights=True)],
    )

    ################################################################################
    # Evaluate model
    ################################################################################
    print("Testing model")
    loss, acc = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    print("Done. Test loss: {}. Test acc: {}".format(loss, acc))