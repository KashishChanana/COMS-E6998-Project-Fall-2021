layers_list = [2, 1]
units_list = [256, 128]
dropout_list = [0.2, 0.1]
batch_size_list = [16, 8]

import time
import numpy as np
import pandas as pd
import gzip
import tensorflow as tf
import sklearn.model_selection
import os


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


class TimeHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []
    
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


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


def create_cnn_transformer_model(layers, units, dropout, input_shape=(10, 224, 224, 3)):

    resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape[1:])

    model = tf.keras.Sequential([
        resnet50,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    model.trainable = False
    
    ip = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.TimeDistributed(model)(ip)

    x = TransformerEncoder(layers, input_shape[0], 4, units, units, dropout)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    op = tf.keras.layers.Dense(101, activation='softmax')(x)

    return tf.keras.Model(inputs=[ip], outputs=[op])


def train(layers, units, dropout, batch_size, logs_folder):

    print('Model training: ' + 'transformer_' + str(layers) + '_' + str(units) + '_' + str(dropout) + '_' + str(batch_size))

    train_data_gen = DataGenerator(list(train_path), list(train_label), batch_size=batch_size, shuffle=True)
    val_data_gen = DataGenerator(list(val_path), list(val_label), batch_size=batch_size, shuffle=True)

    cnn_transformer_model = create_cnn_transformer_model(layers, units, dropout, input_shape=(10, 224, 224, 3))

    cnn_transformer_model.summary()

    cnn_transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                  loss='categorical_crossentropy', metrics='accuracy')

    time_history = TimeHistory()

    history = cnn_transformer_model.fit(train_data_gen, epochs=10, validation_data=val_data_gen, callbacks=[time_history])
    history.history['time_history'] = time_history.times
    
    csv_name = 'transformer_' + str(layers) + '_' + str(units) + '_' + str(dropout) + '_' + str(batch_size) + '_' + str(cnn_transformer_model.count_params()) + '.csv'
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(logs_folder, csv_name))


df = pd.read_csv('train.csv')

train_path, val_path, train_label, val_label = sklearn.model_selection.train_test_split(
    df['path'],
    pd.get_dummies(df['label']).values,
    test_size=0.1,
    stratify=df['label'],
    random_state=42
)

logs_folder = 'logs/transformer'
os.makedirs(logs_folder, exist_ok=True)

for layers in layers_list:
    for units in units_list:
        for dropout in dropout_list:
            for batch_size in batch_size_list:
                try:
                    train(layers, units, dropout, batch_size, logs_folder)
                    print('Trained and saved: ' + 'transformer_' + str(layers) + '_' + str(units) + '_' + str(dropout) + '_' + str(batch_size))
                except:
                    print('Failed to train: ' + 'transformer_' + str(layers) + '_' + str(units) + '_' + str(dropout) + '_' + str(batch_size))