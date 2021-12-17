import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, path, label, batch_size=4, shuffle=True):
        self.batch_size = batch_size
        self.indices = [i for i in range(len(path))]
        self.shuffle = shuffle
        self.path = path
        self.label = label
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        idx_list = self.idx[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.indices[k] for k in idx_list]
        X, y = self.get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.idx = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.idx)

    def get_data(self, batch):
        X = [self.path[i] for i in batch]
        y = [self.label[i] for i in batch]
        X_res = []
        y_res = []
        for i, d in enumerate(zip(X, y)):
            X_res.append(np.load(gzip.GzipFile(d[0], "r")))
            y_res.append(d[1])
        X_res = np.array(X_res).astype(np.float32)
        y_res = np.array(y_res).astype(np.float32)
        return X_res, y_res
