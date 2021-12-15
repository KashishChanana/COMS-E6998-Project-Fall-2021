import tensorflow as tf


def ffn(dim_model, dim_feedforward, dropout):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dim_feedforward, activation=tf.keras.layers.PReLU()),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(dim_model)
    ])

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, dim_model, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, dim_model, dropout=dropout)
        self.ffn = ffn(dim_model, dim_feedforward, dropout)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, src):

        x = self.layernorm1(src)
        x = self.mha(x, x)
        x = self.dropout1(x)
        x1 = x + src
        
        x = self.layernorm2(x1)
        x = self.ffn(x)
        x = self.dropout2(x)
        x2 = x + x1

        return x2

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, time_features, dim_model):
        super(PositionalEncoding, self).__init__()

        self.time_features = time_features
        self.projection = tf.keras.layers.Dense(dim_model)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=time_features, output_dim=dim_model
        )

    def call(self, src, n):
    
        positions = tf.range(start=0, limit=self.time_features, delta=1)
        src_pos = self.projection(src) + self.position_embedding(positions)

        return src_pos

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, time_features, num_heads, dim_model, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.time_features = time_features

        self.positional_encoding = PositionalEncoding(time_features, dim_model)
        self.enc_layers = [TransformerEncoderLayer(num_heads, dim_model, dim_feedforward, dropout) for _ in range(num_layers)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, src):

        src_pos = self.positional_encoding(src, self.time_features)
        for n in range(self.num_layers):
            src_pos = self.enc_layers[n](src_pos)

        return src_pos

import pandas as pd
import numpy as np
import gzip
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

def create_cnn_transformer_model(input_shape):

    resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
    model = tf.keras.Sequential([resnet50, tf.keras.layers.GlobalAveragePooling2D()])
    model.trainable = False
    
    ip = tf.keras.layers.Input(shape=input_shape)
    cnn = tf.keras.layers.TimeDistributed(model)(ip)
    transformer = TransformerEncoder(2, input_shape[0], 4, 512, 2048, 0.1)(cnn)
    average = tf.keras.layers.GlobalAveragePooling1D()(transformer)
    #dense = tf.keras.layers.Dense(256, activation="relu")(average)
    op = tf.keras.layers.Dense(101, activation='softmax')(dense)

    return tf.keras.Model(inputs=[ip], outputs=[op])

df = pd.read_csv('train.csv')
train_path, val_path, train_label, val_label = train_test_split(
    df['path'],
    pd.get_dummies(df['label']).values,
    test_size=0.1,
    stratify=df['label'],
    random_state=42
)

train_data_gen = DataGenerator(list(train_path), list(train_label), shuffle=True)
val_data_gen = DataGenerator(list(val_path), list(val_label), shuffle=True)

cnn_lstm_model = create_cnn_transformer_model((20, 224, 224, 3))

cnn_lstm_model.summary()

cnn_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
loss='categorical_crossentropy', metrics='accuracy')

cnn_lstm_model.fit(train_data_gen, epochs=20, validation_data=val_data_gen)