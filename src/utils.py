
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
       'ResultSpeed']

DUMMY_COLUMN, TARGET_VARIABLE = 'Species', 'WnvPresent'

TRAP_DICT_FILE = "../output/trap_dict"

def process_spray_data():
    _weather = pd.read_csv("../input/spray.csv")[WEATHER_COLS]

def process_weather_data():
    _weather = pd.read_csv("../input/weather.csv")[WEATHER_COLS]
    _weather = _weather.sort_values("Date",ascending=True)
    # Format precipitation and pressure.
    # For precipation, set missing and traces values to 0.0
    # Fill missing pressure values with average (3 in total)
    convert_precipation = lambda x : float(x) if x not in ['  T','M'] else 0.
    convert_pressure = lambda x : float(x) if x not in ['  T','M'] else np.nan
    _weather['PrecipTotal'] = _weather['PrecipTotal'].apply(convert_precipation)
    # Average Precip Over 5 days
    r = _weather.PrecipTotal.rolling(window=5)
    _weather['AvgPrecip'] = r.mean().fillna(0)
    _weather['StnPressure'] = _weather['StnPressure'].apply(convert_pressure)
    _weather['StnPressure'] = _weather['StnPressure'].fillna(_weather['StnPressure'].mean())
    # Drop missing wet-bulb values (n = 3)
    # There are two stations and we still have the other stations
    _weather['WetBulb'] = _weather['WetBulb'][_weather['WetBulb'] != 'M'].astype(float)
    # Drop missing average temperatures (n = 10)
    # There are two stations and we still have the other stations
    _weather['Tavg'] = _weather['Tavg'][_weather['Tavg'] != 'M'].astype(float)
    # Calculate the difference between wetBulb and Avg Tem
    # Trail over last 5 days
    _weather['Diff'] = _weather['Tavg'] - _weather['DewPoint']
    r = _weather.Diff.rolling(window=5)
    _weather['Diff'] = r.mean().fillna(np.mean(_weather['Diff']))
    # Average out two stations
    _weather = _weather.groupby(['Date']).mean().reset_index()
    # Month and Week
    get_month = lambda d: float(d.split('-')[1])
    get_week = lambda d: round(int(d.split('-')[1]) * 4 + int(d.split('-')[2]) / 7)
    _weather['week'] = _weather['Date'].apply(get_week)
    _weather['month'] = _weather['Date'].apply(get_month)
    return _weather


def get_train_or_test_data(train=True):
    print("Processing weather data...")
    weather = process_weather_data()


    def get_trap_feature(traps,trap_stats):
        _trap_feature = []
        for trap in traps:
            if trap in trap_stats:
                _trap_feature.append(trap_stats[trap])
            else:
                _trap_feature.append(trap_stats['mean'])
        return _trap_feature

    if train:
        print("Processing training data...")
        dt = pd.read_csv("../input/train.csv")[TRAIN_COLS]
        # Approximate the number of mosquitos found in traps
        # so that we can lookup at test time
        def _compute_trap_stats(dt):
            trap_stats = dt.groupby('Trap').agg([np.mean,len])['NumMosquitos'].reset_index().sort_values('mean',ascending=False)
            trap_stats.index = trap_stats.Trap
            trap_stats = trap_stats.drop(["len","Trap"],axis=1)
            trap_dict = trap_stats.to_dict()['mean']
            _mean = np.mean(np.array(list(trap_dict.values())).flatten())
            trap_dict['mean'] = _mean
            return trap_dict

        trap_stats = _compute_trap_stats(dt)
        traps = dt.Trap.tolist()
        dt['MosquitoByTrap'] = get_trap_feature(traps, trap_stats)
        # Drop Trap Info
        dt = dt.drop(['NumMosquitos','Trap'],axis=1)
        # Persist hash table for lookup at test time
        _file = open(TRAP_DICT_FILE,'wb')
        pickle.dump(trap_stats,_file)
        _file.close()
    else:
        print("Processing test data...")
        dt = pd.read_csv("../input/test.csv")[TEST_COLS]
        # Populate trap counts
        _file = open(TRAP_DICT_FILE,'rb')
        # load the object from the file
        trap_stats = pickle.load(_file)
        _file.close()
        traps = dt.Trap.tolist()
        dt['MosquitoByTrap'] = get_trap_feature(traps, trap_stats)
        dt = dt.drop(['Trap'],axis=1)
    # Merge
    dt = dt.merge(weather,on='Date',how='inner')
    # Drop the date
    dt = dt.drop('Date',axis=1)
    return dt

def preprocess_data(X, train=True, scaler=None):
    # Target variable
    if train:
        Y = np_utils.to_categorical(X[TARGET_VARIABLE])
        X = X.drop(TARGET_VARIABLE, axis=1)
    else:
        Y = None

    # One Hot Encoding
    dummy = pd.get_dummies(X[DUMMY_COLUMN]).astype(float)
    X = X.drop(DUMMY_COLUMN, axis=1)
    if train:
        # index only present in test data.
        # appending here so that dims match at test time
        dummy['UNSPECIFIED CULEX'] = 0.
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
