# defining the hyperparameters for experimentation
layers_list = [2, 1]
units_list = [256, 128]
dropout_list = [0.2, 0.1]
batch_size_list = [16, 8]

# importing libraries
import time
import numpy as np
import pandas as pd
import gzip
import tensorflow as tf
import sklearn.model_selection
import os
from time_history_callback import *
from data_flow import *

# class creating the entire CNN-LSTM Architecture 
def create_cnn_lstm_model(layers, units, dropout, input_shape=(10, 224, 224, 3)):

    resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape[1:])

    model = tf.keras.Sequential([
        resnet50,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    model.trainable = False
    
    ip = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.TimeDistributed(model)(ip)

    for _ in range(layers-1):
        x = tf.keras.layers.LSTM(units, return_sequences=True, dropout=dropout)(x)
    x = tf.keras.layers.LSTM(units, dropout=dropout)(x)
    
    op = tf.keras.layers.Dense(101, activation='softmax')(x)

    return tf.keras.Model(inputs=[ip], outputs=[op])

# training the model
def train(layers, units, dropout, batch_size, logs_folder):

    print('Model training: ' + 'lstm_' + str(layers) + '_' + str(units) + '_' + str(dropout) + '_' + str(batch_size))

    train_data_gen = DataGenerator(list(train_path), list(train_label), batch_size=batch_size, shuffle=True)
    val_data_gen = DataGenerator(list(val_path), list(val_label), batch_size=batch_size, shuffle=True)

    cnn_lstm_model = create_cnn_lstm_model(layers, units, dropout, input_shape=(10, 224, 224, 3))

    cnn_lstm_model.summary()

    cnn_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    time_history = TimeHistory()

    history = cnn_lstm_model.fit(train_data_gen, epochs=10, validation_data=val_data_gen, callbacks=[time_history])
    history.history['time_history'] = time_history.times
    
    csv_name = 'lstm_' + str(layers) + '_' + str(units) + '_' + str(dropout) + '_' + str(batch_size) + '_' + str(cnn_lstm_model.count_params()) + '.csv'
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

# generating logs for the CNN-LSTM architecture
logs_folder = 'logs/lstm'
os.makedirs(logs_folder, exist_ok=True)

for layers in layers_list:
    for units in units_list:
        for dropout in dropout_list:
            for batch_size in batch_size_list:
                try:
                    train(layers, units, dropout, batch_size, logs_folder)
                    print('Trained and saved: ' + 'lstm_' + str(layers) + '_' + str(units) + '_' + str(dropout) + '_' + str(batch_size))
                except:
                    print('Failed to train: ' + 'lstm_' + str(layers) + '_' + str(units) + '_' + str(dropout) + '_' + str(batch_size))