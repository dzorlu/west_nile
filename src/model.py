#!/usr/bin/python3

import pandas as pd
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import KFold
from keras.regularizers import l2

from sklearn import metrics
from datetime import datetime

from utils import *

NUM_FOLDS = 4

def build_model(input_dim, output_dim):
    """Creates a three layer Keras NN model"""
    print("Building the model...")
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-04))

    return model

def train_model(X,Y):
    av_roc = 0.
    model = build_model(X.shape[1], Y.shape[1])
    print("Training model with {} folds...".format(NUM_FOLDS))
    kf = KFold(n_splits=NUM_FOLDS)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]
        model.fit(X_train, y_train, nb_epoch=250, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        valid_preds = model.predict_proba(X_val,verbose=0)
        roc = metrics.roc_auc_score(y_val, valid_preds)
        print("ROC:", roc)
        av_roc += roc
    print("***")
    print('Average ROC:', av_roc/NUM_FOLDS)
    print("Saving model...")
    _timestamp = datetime.now().strftime("%m%d%H%M")
    model.save("../saved_models/model_{}.h5".format(_timestamp))
    return model

def save_predictions(preds):
    preds = pd.DataFrame(preds[:,1], columns=["WnvPresent"])
    preds['Id'] = preds.index + 1
    preds = preds[['Id','WnvPresent']]
    _timestamp = datetime.now().strftime("%m%d%H%M")
    print("Saving prediction")
    pd.DataFrame.to_csv(preds,"../output/output_{}.csv".format(_timestamp),index=False)

def run():
    # Prepare data`
    train_data = get_train_or_test_data()
    X, Y, scaler = preprocess_data(train_data)

    # Fit
    fitted_model = train_model(X, Y)

    # Predict
    test_data = get_train_or_test_data(train=False)
    X_test, _, _ = preprocess_data(test_data, train=False, scaler=scaler)

    preds = fitted_model.predict_proba(X_test, verbose=0)
    save_predictions(preds)

if __name__ == '__main__':
    print("run")
    run()
