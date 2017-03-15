# West Nile Virus Detection

Given weather, location, testing, and spraying data, the pipeline attempts to identify when and where different species of mosquitos will test positive for West Nile virus. For more information, please see the [Kaggle](https://www.kaggle.com/c/predict-west-nile-virus) page.

Most seamless way to run the script is using a [Docker](https://github.com/udacity/CarND-Term1-Starter-Kit) image containing required libraries. Please run the following on the command line to start the iPython notebook:

```
$ docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

The project currently lives in two files. [`utils.py`](https://github.com/dzorlu/west_nile/blob/master/src/utils.py) contains the pre-processing logic. [`model.py`](https://github.com/dzorlu/west_nile/blob/master/src/model.py) is used to build and train the model and to make predictions. Particularly, the pipeline can be succinctly summarized with the following [lines](https://github.com/dzorlu/west_nile/blob/master/src/model.py#L64). The output is a CSV file that contains the probability of a positive result for each line in the test dataset.

```
# Prepare data
train_data = get_train_or_test_data()
X, Y, scaler = preprocess_data(train_data)

# Fit
fitted_model = train_model(X, Y)

# Predict
test_data = get_train_or_test_data(train=False)
X_test, _, _ = preprocess_data(test_data, train=False, scaler=scaler)

preds = fitted_model.predict_proba(X_test, verbose=0)
save_predictions(preds)
```
