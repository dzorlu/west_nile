# West Nile virus detection

Given weather, location, testing, and spraying data, the pipeline summarized below attempts to identify when and where different species of mosquitos will test positive for West Nile virus. For more information, please see the ![Kaggle](https://www.kaggle.com/c/predict-west-nile-virus) page.

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

Most seamless way to run the script is using a ![Docker](https://github.com/udacity/CarND-Term1-Starter-Kit) image containing required libraries. Please run the following on the command line to start the iPython notebook:

```
$ docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```
