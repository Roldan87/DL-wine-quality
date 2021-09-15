import pandas as pd
import numpy as np
from scipy.sparse.sputils import matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, plot_confusion_matrix, accuracy_score, r2_score
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def read_dataset(path):
    df = pd.read_csv(path, index_col=0)
    df = df.drop_duplicates()
    df.reset_index(drop=True)
    return df

# COACHES' NOTES: would it not make more sense to give a threshold as an argument, and then use a lambda to map the df?
def define_wine_quality(df):
    good_wine = [7,8,9]
    bad_wine = [3,4,5,6]
    df.quality = df.quality.replace(to_replace=good_wine, value=1)
    df.quality = df.quality.replace(to_replace= bad_wine, value=0)
    return df

def resampling(df):
    df_good_wine = df[df['quality'] == 1]
    df_bad_wine = df[df['quality'] == 0]
    # COACHES' NOTES: unlucky number 13, why was this not just an argument? Or better yet; calculated from the ratio between bad and good wines?
    for i in range(13):
        df = df.append(df_good_wine)
    return df

def featuring_data(df):
    features = df.drop(columns=['quality'], axis=1)
    target = df['quality']
    return features, target

def split_standardize_data(features, target):
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, stratify=target)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test

# COACHES' NOTES: What is this used for? You'd be better off trusting sklearn's cross val functions.
def validation_data(x_train, y_train):
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = y_train[:1000]
    partial_y_train = y_train[1000:]
    return x_val, partial_x_train, y_val, partial_y_train

# COACHES' NOTES: I feel as if you could just provide X, y and use your other functions to split them in this function. Also, add arguments for layer sizes, learning rates... so you can use this function for tuning later.
def build_network(x_val, partial_x_train, y_val, partial_y_train, x_test, y_test):
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(1024, input_dim=11, kernel_initializer='normal', activation='tanh'))
    model.add(layers.Dense(512, kernel_initializer='normal', activation='tanh'))
    model.add(layers.Dense(64, kernel_initializer='normal', activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    history = model.fit(partial_x_train, partial_y_train, 
                        epochs=100, batch_size=256, verbose=1, 
                        validation_data=(x_val, y_val), 
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
    train_loss, train_acc = model.evaluate(partial_x_train, partial_y_train)
    print('train_acc: ', train_acc)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('test_acc: ', test_acc)
    prediction = model.predict(x_test)
    cm = tf.math.confusion_matrix(y_test, prediction)
    return history, cm

def visualize_results(history, matrix):
    history_dict = history.history
    acc = history_dict['accuracy']
    loss = history_dict['loss']
    epochs = range(1, len(acc) + 1)
    #Loss-Acc curves
    plt.plot(epochs, acc, 'bo', label='Accuracy')
    plt.plot(epochs, loss, 'b', label='Loss')
    plt.title('Training loss-accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc-Loss')
    plt.legend()
    plt.show()
    #Confusion Matrix
    ax= plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_title('Predictions Confusion Matrix') 
    ax.xaxis.set_ticklabels(['bad wine', 'good wine']) 
    ax.yaxis.set_ticklabels(['bad wine', 'good wine'])
    plt.show()


#Function Calls

df_wine = read_dataset('./assets/wine.csv')
df_wine = define_wine_quality(df_wine)
df_wine = resampling(df_wine)
df_wine = df_wine.sample(frac=1, random_state=1)    #Shuffle data
features, target = featuring_data(df_wine)
x_train, x_test, y_train, y_test = split_standardize_data(features, target)
x_val, partial_x_train, y_val, partial_y_train = validation_data(x_train, y_train)
history, cm = build_network(x_val, partial_x_train, y_val, partial_y_train, x_test, y_test)
visualize_results(history, cm)


# COACHES' NOTES: Overall, pretty good, but lacking in typing and return value declaration. 