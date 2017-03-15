
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from itertools import product

import pickle

TRAIN_COLS = ['Date', 'Species',
       'Latitude', 'Longitude',
       'Trap','NumMosquitos','WnvPresent']

TEST_COLS = ['Date', 'Species',
        'Trap','Latitude', 'Longitude']

WEATHER_COLS = ['Date','Tmax', 'Tmin',
       'Tavg', 'DewPoint',
       'WetBulb', 'StnPressure','PrecipTotal',
       'ResultSpeed','SeaLevel']

DUMMY_COLUMNS, TARGET_VARIABLE = ['month','Species'], 'WnvPresent'

KNN_FILE = "../output/knn"
NUM_LOOKBACK_DAYS = 10

def process_spray_data():
    _weather = pd.read_csv("../input/spray.csv")[WEATHER_COLS]

def process_weather_data():
    _weather = pd.read_csv("../input/weather.csv")[WEATHER_COLS]
    _weather = _weather.sort_values("Date",ascending=True)
    # Format precipitation and pressure.
    # For precipation, set missing and traces values to 0.0
    # Fill missing pressure values with average (3 in total)
    convert_sealevel = lambda x : float(x) if x not in ['  T','M'] else np.nan
    convert_precipation = lambda x : float(x) if x not in ['  T','M'] else 0.
    convert_pressure = lambda x : float(x) if x not in ['  T','M'] else np.nan
    _weather['PrecipTotal'] = _weather['PrecipTotal'].apply(convert_precipation)
    # Average Precip Over NUM_LOOKBACK_DAYS days
    r = _weather.PrecipTotal.rolling(window=NUM_LOOKBACK_DAYS)
    _weather['AvgPrecip'] = r.mean().fillna(0)
    _weather['StnPressure'] = _weather['StnPressure'].apply(convert_pressure)
    _weather['StnPressure'] = _weather['StnPressure'].fillna(_weather['StnPressure'].mean())
    # SeaLevel Adjustment
    _weather['SeaLevel'] = _weather['SeaLevel'].apply(convert_sealevel)
    _weather['SeaLevel'] = _weather['SeaLevel'].fillna(_weather['SeaLevel'].mean())
    # Drop missing wet-bulb values (n = 3)
    # There are two stations and we still have the other stations
    _weather['WetBulb'] = _weather['WetBulb'][_weather['WetBulb'] != 'M'].astype(float)
    # Drop missing average temperatures (n = 10)
    # There are two stations and we still have the other stations
    _weather['Tavg'] = _weather['Tavg'][_weather['Tavg'] != 'M'].astype(float)
    # Calculate the difference between wetBulb and Avg Tem
    # Trail over last NUM_LOOKBACK_DAYS days
    _weather['Diff'] = _weather['Tavg'] - _weather['DewPoint']
    r = _weather.Diff.rolling(window=NUM_LOOKBACK_DAYS)
    _weather['Diff'] = r.mean().fillna(np.mean(_weather['Diff']))
    # Average out two stations
    _weather = _weather.groupby(['Date']).mean().reset_index()
    # Month and Week
    get_month = lambda d: (d.split('-')[1])
    get_week = lambda d: int(int(d.split('-')[1]) * 4 + int(d.split('-')[2]) / 7)
    _weather['week'] = _weather['Date'].apply(get_week)
    # Month as a categorical value
    _weather['month'] = _weather['Date'].apply(get_month).astype('category')
    return _weather


def get_train_or_test_data(train=True):
    def _fit_neighborhood_model(dt):
        _temp = dt.copy()
        """
        Fit a KD-Tree for each month
        given (lon,lat) and mosquito counts
        """
        get_month = lambda d: int(d.split('-')[1])
        _temp['month'] = _temp['Date'].apply(get_month)
        months = set(_temp.month.astype(int).tolist())
        neighborhood_model = {}
        for _month in months:
            neigh = KNeighborsRegressor(n_neighbors=3)
            _data = _temp[['Longitude','Latitude','NumMosquitos','month']][_temp['month'] == _month]
            _data = _data.groupby(['Longitude','Latitude']).mean()['NumMosquitos'].reset_index()
            _X, _y = np.array(_data[['Longitude','Latitude']]), _data['NumMosquitos']
            neigh.fit(_X, _y)
            neighborhood_model[_month] = neigh
        return neighborhood_model

    def _get_mosquito_bias(dt, model):
        _temp = dt.copy()
        get_month = lambda d: int(d.split('-')[1])
        _temp['month'] = _temp['Date'].apply(get_month)
        # Compute the approx mosquito count
        def compute_bias(row):
            _x = np.array([row.Longitude,row.Latitude]).reshape(1,-1)
            return model[row.month].predict(_x)[0]
        return _temp.apply(compute_bias,axis=1)

    print("Processing weather data...")
    weather = process_weather_data()
    if train:
        print("Processing training data...")
        dt = pd.read_csv("../input/train.csv")[TRAIN_COLS]
        # Approximate the number of mosquitos found in traps
        # so that we can lookup at test time
        neigh = _fit_neighborhood_model(dt)
        dt['MosquitoBias'] = _get_mosquito_bias(dt, neigh)
        dt = dt.drop(['NumMosquitos','Trap'],axis=1)
        # Persist model for lookup at test time
        _file = open(KNN_FILE,'wb')
        pickle.dump(neigh, _file)
        _file.close()
    else:
        print("Processing test data...")
        dt = pd.read_csv("../input/test.csv")[TEST_COLS]
        # Populate trap counts
        _file = open(KNN_FILE,'rb')
        # load the object from the file
        neigh = pickle.load(_file)
        _file.close()
        traps = dt.Trap.tolist()
        dt['MosquitoBias'] = _get_mosquito_bias(dt, neigh)
        dt = dt.drop(['Trap'],axis=1)
    # Merge
    dt = dt.merge(weather,on='Date',how='inner')
    # Drop the date
    dt = dt.drop(['Date'],axis=1)
    return dt

def preprocess_data(X, train=True, scaler=None):
    # Target variable
    if train:
        Y = np_utils.to_categorical(X[TARGET_VARIABLE])
        X = X.drop(TARGET_VARIABLE, axis=1)
    else:
        Y = None

    # One Hot Encoding
    dummy = pd.DataFrame()
    for dummy_column in DUMMY_COLUMNS:
        _temp = pd.get_dummies(X[dummy_column]).astype(float)
        dummy = pd.concat([dummy,_temp],axis=1)

    X = X.drop(DUMMY_COLUMNS, axis=1)
    if train:
        # index only present in test data.
        # appending here so that dims match at test time
        dummy['UNSPECIFIED CULEX'] = 0.
        # drop month 5 as it is not present in test data
        dummy = dummy.drop("05",axis=1)
    X = pd.concat([X,dummy],axis=1)

    # Standardize features by removing the mean and scaling to unit variance
    if not scaler:
        scaler = StandardScaler()
        print("training data features {}".format(X.columns.tolist()))
        scaler.fit(X)
    else:
        print("test data features {}".format(X.columns.tolist()))
    X = scaler.transform(X)
    return X, Y, scaler
