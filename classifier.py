from __future__ import division

import numpy as np
import sys
import pynn
from sklearn import (metrics, cross_validation, linear_model,
        preprocessing, ensemble, svm)

SEED = 111  # always use a seed for randomized procedures


def load_data(filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s. 
    """

    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open("data/" + filename), delimiter=',',
                      usecols=range(1, 9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open("data/" + filename), delimiter=',',
                            usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


def run_model(test, lgr_args, rf_args, svm_args, mix):
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.

    if test is true, it runs a 10-fold cross validation and reports the mean
    AUC. If not, it trains the model on the whole dataset and outputs a prediction
    file.
    """
    lgr_model = linear_model.LogisticRegression(**lgr_args)
    rf_model = ensemble.RandomForestRegressor(**rf_args)
    svm_model = svm.SVC(probability=True, **svm_args)

    # === load data in memory === #
    print("loading data")
    y, X = load_data("train.csv")
    y_test, X_test = load_data("test.csv", use_labels=False)

    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    oneHotencoder = preprocessing.OneHotEncoder()
    oneHotencoder.fit(np.vstack((X, X_test)))

    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(np.vstack((X, X_test)))

    X_sparse = oneHotencoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test_sparse = oneHotencoder.transform(X_test)

    X_label = labelEncoder.transform(X)
    X_test_label = labelEncoder.transform(X_test)

    # if you want to create new features, you'll need to compute them
    # before the encoding, and append them to your dataset after

    if test:
        # === training & metrics === #
        mean_auc = 0.0
        n = 10  # repeat the CV procedure 10 times to get more precise results
        for i in range(n):
            # for each iteration, randomly hold out 20% of the data as CV set
            X_train_label, X_cv_label, y_train, y_cv = cross_validation.train_test_split(
                X_label, y, test_size=.20, random_state=i*SEED)

            X_train_sparse, X_cv_sparse, y_train, y_cv = cross_validation.train_test_split(
                X_sparse, y, test_size=.20, random_state=i*SEED)

            # if you want to perform feature selection / hyperparameter
            # optimization, this is where you want to do it

            # train model and make predictions
            print("Training models")
            lgr_model.fit(X_train_sparse, y_train)
            rf_model.fit(X_train_label, y_train)
            svm_model.fit(X_train_sparse, y_train)

            lgr_preds = lgr_model.predict_proba(X_cv_sparse)[:, 1]
            rf_preds = rf_model.predict(X_cv_label)
            svm_preds = svm_model.predict_proba(X_cv_sparse)[:, 1]

            # normalise the mix ratios, then mix the predictor outcomes
            mix = np.array(mix, dtype="float64")
            mix /= mix.sum()
            combined_preds = mix[0]*lgr_preds + mix[1]*rf_preds + mix[2]*svm_preds

            # compute AUC metric for this CV fold
            fpr, tpr, thresholds = metrics.roc_curve(y_cv, lgr_preds)
            roc_auc = metrics.auc(fpr, tpr)
            print("Logistic Regression AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc))

            fpr, tpr, thresholds = metrics.roc_curve(y_cv, svm_preds)
            roc_auc = metrics.auc(fpr, tpr)
            print("SVM Regression AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc))

            fpr, tpr, thresholds = metrics.roc_curve(y_cv, rf_preds)
            roc_auc = metrics.auc(fpr, tpr)
            print("RF Regression AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc))

            fpr, tpr, thresholds = metrics.roc_curve(y_cv, combined_preds)
            roc_auc = metrics.auc(fpr, tpr)
            print("Combined AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc))

            mean_auc += roc_auc
        return mean_auc/n

    else: 
        # === Predictions === #
        # When making predictions, retrain the model on the whole training set
        lgr_model.fit(X_sparse, y)
        svm_model.fit(X_sparse, y)
        rf_model.fit(X_label, y) 
        
        lgr_preds = lgr_model.predict_proba(X_test_sparse)[:, 1]
        svm_preds = svm_model.predict(X_test_sparse)
        rf_preds = rf_model.predict(X_test_label)

        # normalise the mix ratios, then mix the predictor outcomes
        mix = np.array(mix, dtype="float64")
        mix /= mix.sum()
        combined_preds = mix[0]*lgr_preds + mix[1]*rf_preds + mix[2]*svm_preds

        filename = raw_input("Enter name for submission file: ")
        save_results(combined_preds, filename + ".csv")
        return 0

if __name__ == '__main__':
    # (lgr_args, rf_args, svm_args, mix)
    param_sets = [({"C":3},
                   {"n_estimators":100, "min_samples_split":8, "min_samples_leaf":1},
                   {"C":1, "gamma":0.1, "cache_size":500},
                   [9.0, 6.0, 6.0]), # Mean AUC: 0.870934
                  ({"C":3},
                   {"n_estimators":100, "min_samples_split":8, "min_samples_leaf":1},
                   {"C":1, "gamma":0.1, "cache_size":500},
                   [9.0, 6.0, 3.0]), # Mean AUC: 0.870739
                  ({"C":3},
                   {"n_estimators":100, "min_samples_split":8, "min_samples_leaf":1},
                   {"C":1, "gamma":0.1, "cache_size":500},
                   [6.0, 4.0, 0.0]), # Mean AUC: 0.869819
                  ({"C":3},
                   {"n_estimators":100, "min_samples_split":8, "min_samples_leaf":1},
                   {"C":1, "gamma":0.1, "cache_size":500},
                   [9.0, 3.0, 6.0]), # Mean AUC: 0.869334
                  ({"C":3},
                   {"n_estimators":100, "min_samples_split":8, "min_samples_leaf":1},
                   {"C":1, "gamma":0.1, "cache_size":500},
                   [6.0, 0.0, 4.0]) # Mean AUC: 0.860840
                  ]
    if sys.argv[1] == "test":
        for params in param_sets:
            print("lgr_args:{} rf_args:{} svm_args:{} mix:{}".format(*params))
            mean_auc = run_model(True, *params)
            print("Mean AUC: %f" % (mean_auc))
    elif sys.argv[1] == "run":
        params = param_sets[0]
        print("lgr_args:{} rf_args:{} svm_args:{} mix:{}".format(*params))
        mean_auc = run_model(False, *params)
