import pandas as pd
import numpy as np
import gzip
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

def create_cnn_lstm_model(input_shape):

    resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = tf.keras.Sequential([resnet50, tf.keras.layers.GlobalAveragePooling2D()])
    model.trainable = False
    
    ip = tf.keras.layers.Input(shape=input_shape)
    cnn = tf.keras.layers.TimeDistributed(model)(ip)
    lstm_1 = tf.keras.layers.LSTM(512, return_sequences=True, dropout=0.1)(cnn)
    lstm_2 = tf.keras.layers.LSTM(512, dropout=0.1)(lstm_1)
    dense = tf.keras.layers.Dense(256, activation="relu")(lstm_2)
    op = tf.keras.layers.Dense(101, activation='softmax')(dense)

    return tf.keras.Model(inputs=[ip], outputs=[op])

df = pd.read_csv('train.csv')

from sklearn.model_selection import train_test_split
train_path, val_path, train_label, val_label = train_test_split(
    df['path'],
    pd.get_dummies(df['label']).values,
    test_size=0.1,
    stratify=df['label'],
    random_state=42
)

train_data_gen = DataGenerator(list(train_path), list(train_label), batch_size=4, shuffle=True)
val_data_gen = DataGenerator(list(val_path), list(val_label), batch_size=4, shuffle=True)

cnn_lstm_model = create_cnn_lstm_model((20, 224, 224, 3))

cnn_lstm_model.summary()

cnn_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
loss='categorical_crossentropy', metrics='accuracy')

cnn_lstm_model.fit(train_data_gen, epochs=20, validation_data=val_data_gen)