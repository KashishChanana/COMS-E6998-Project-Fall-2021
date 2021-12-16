time_frame_list = [4, 3, 2]
dropout_list = [0.2, 0.1]
batch_size_list = [16, 8]


import time
import numpy as np
import pandas as pd
import gzip
import tensorflow as tf
import sklearn.model_selection
import os


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


def create_3d_cnn_model(time_frame, dropout, padding='same', activation='relu', input_shape=(10, 224, 224, 3)):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv3D(32, (time_frame, 3, 3), padding=padding, activation=activation))
    model.add(tf.keras.layers.AveragePooling3D((1, 2, 2)))
    model.add(tf.keras.layers.SpatialDropout3D(dropout))

    model.add(tf.keras.layers.BatchNormalization())    
    model.add(tf.keras.layers.Conv3D(64, (time_frame, 3, 3), padding=padding, activation=activation))
    model.add(tf.keras.layers.AveragePooling3D((1, 2, 2)))
    model.add(tf.keras.layers.SpatialDropout3D(dropout))

    model.add(tf.keras.layers.BatchNormalization())    
    model.add(tf.keras.layers.Conv3D(128, (time_frame, 3, 3), padding=padding, activation=activation))
    model.add(tf.keras.layers.AveragePooling3D((1, 2, 2)))
    model.add(tf.keras.layers.SpatialDropout3D(dropout))

    model.add(tf.keras.layers.BatchNormalization())    
    model.add(tf.keras.layers.Conv3D(256, (time_frame, 3, 3), padding=padding, activation=activation))
    model.add(tf.keras.layers.AveragePooling3D((1, 2, 2)))
    model.add(tf.keras.layers.SpatialDropout3D(dropout))

    model.add(tf.keras.layers.BatchNormalization())    
    model.add(tf.keras.layers.Conv3D(512, (time_frame, 3, 3), padding=padding, activation=activation))
    model.add(tf.keras.layers.AveragePooling3D((1, 2, 2)))
    model.add(tf.keras.layers.SpatialDropout3D(dropout))

    model.add(tf.keras.layers.AveragePooling3D(pool_size=(1, 7, 7)))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(101, activation='softmax'))
    
    return model


def train(time_frame, dropout, batch_size, logs_folder):

    print('Model training: ' + 'cnn_' + str(time_frame) + '_' + str(dropout) + '_' + str(batch_size))

    train_data_gen = DataGenerator(list(train_path), list(train_label), batch_size=batch_size, shuffle=True)
    val_data_gen = DataGenerator(list(val_path), list(val_label), batch_size=batch_size, shuffle=True)

    cnn_transformer_model = create_3d_cnn_model(time_frame, dropout, input_shape=(10, 224, 224, 3))

    cnn_transformer_model.summary()

    cnn_transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                  loss='categorical_crossentropy', metrics='accuracy')

    time_history = TimeHistory()

    history = cnn_transformer_model.fit(train_data_gen, epochs=10, validation_data=val_data_gen, callbacks=[time_history])
    history.history['time_history'] = time_history.times
    
    csv_name = 'cnn_' + str(time_frame) + '_' + str(dropout) + '_' + str(batch_size) + '_' + str(cnn_transformer_model.count_params()) + '.csv'
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

logs_folder = 'logs/cnn'
os.makedirs(logs_folder, exist_ok=True)

for time_frame in time_frame_list:
    for dropout in dropout_list:
        for batch_size in batch_size_list:
            try:
                train(time_frame, dropout, batch_size, logs_folder)
                print('Trained and saved: ' + 'cnn_' + str(time_frame) + '_' + str(dropout) + '_' + str(batch_size))
            except:
                print('Failed to train: ' + 'cnn_' + str(time_frame) + '_' + str(dropout) + '_' + str(batch_size))