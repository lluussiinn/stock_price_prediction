import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def prediction(model, original, X_test, scaler, loss = 'mse'):

  prediction = model.predict(X_test)
  prediction = prediction.reshape(prediction.shape[0],1)
  prediction_copies_array = np.repeat(prediction, X_test.shape[2], axis=-1) #change this one

  # pred = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction), X_test.shape[2])))[:,3]
  pred = np.reshape(prediction_copies_array, (len(prediction), X_test.shape[2]))[:, 3]
  if loss == 'mse':
    testScore = mean_squared_error(original, pred)
  if loss == 'mape':
    testScore = mean_absolute_percentage_error(original, pred)

  return pred, testScore



def classification(data, data_main, change):
    if change == 'no change':

        data['Actual_change'] = np.where(data['Close_actual_change'] < data['Close_actual_change'].shift(1), 0, 1)
        data['Pred_change'] = np.where(data['Close_actual_change'] > data['Close_prediction_change'].shift(-1), 0, 1)
        data['Pred_change'] = data.Pred_change.shift(1)
        data = data[1:]
        data['Pred_change'] = data['Pred_change'].astype(int)
        classification_accuracy = len(data[(data.Actual_change == data.Pred_change)]) / len(data)

    elif change == 'absolute':

        data['Actual_change'] = np.where(data['Close_actual_change'] < 0, 0, 1)
        data['Pred_change'] = np.where(data['Close_prediction_change'] < 0, 0, 1)
        data = data[1:]
        data['Pred_change'] = data['Pred_change'].astype(int)
        data['Close_actual'] = data_main.loc['2021-01-01':, 'Close'][30:]
        # data['Close_prediction'] = dataset_test['Close'].shift(1) + data.Close_prediction_change
        classification_accuracy = len(data[(data.Actual_change == data.Pred_change)]) / len(data)

    return data, classification_accuracy

