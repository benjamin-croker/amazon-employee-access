from __future__ import division

import numpy as np
import sys
from sklearn import (metrics, cross_validation, linear_model,
        preprocessing, ensemble)

SEED = 100  # always use a seed for randomized procedures


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


def run_model(test, trees, min_samples_split, min_samples_leaf, C, mix_lgr):
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.

    if test is true, it runs a 10-fold cross validation and reports the mean
    AUC. If not, it trains the model on the whole dataset and outputs a prediction
    file.
    """
    lgr_model = linear_model.LogisticRegression(C=C)
    rf_model = ensemble.RandomForestRegressor(trees, n_jobs=2,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf)

    # === load data in memory === #
    print "loading data"
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
            lgr_model.fit(X_train_sparse, y_train)
            rf_model.fit(X_train_label, y_train)
            
            lgr_preds = lgr_model.predict_proba(X_cv_sparse)[:, 1]
            rf_preds = rf_model.predict(X_cv_label)

            if mix_lgr == "least_certain":
                # pick the prediction with the probability closest to 0.5
                pick_lgr = np.abs(lgr_preds-0.5) < np.abs(rf_preds-0.5)
                combined_preds = pick_lgr*lgr_preds + ~pick_lgr*rf_preds
            elif mix_lgr == "most_certain":
                # pick the prediction with the probability furthest from 0.5
                pick_lgr = np.abs(lgr_preds-0.5) > np.abs(rf_preds-0.5)
                combined_preds = pick_lgr*lgr_preds + ~pick_lgr*rf_preds
            else: 
                combined_preds = mix_lgr*lgr_preds + (1.0 - mix_lgr)*rf_preds

            # compute AUC metric for this CV fold
            fpr, tpr, thresholds = metrics.roc_curve(y_cv, lgr_preds)
            roc_auc = metrics.auc(fpr, tpr)
            print "Logistic Regression AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc)

            fpr, tpr, thresholds = metrics.roc_curve(y_cv, rf_preds)
            roc_auc = metrics.auc(fpr, tpr)
            print "RF Regression AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc)

            fpr, tpr, thresholds = metrics.roc_curve(y_cv, combined_preds)
            roc_auc = metrics.auc(fpr, tpr)
            print "Combined AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc)

            mean_auc += roc_auc
        return mean_auc/n

    else: 
        # === Predictions === #
        # When making predictions, retrain the model on the whole training set
        lgr_model.fit(X_sparse, y) 
        rf_model.fit(X_label, y) 
        
        lgr_preds = lgr_model.predict_proba(X_test_sparse)[:, 1]
        rf_preds = rf_model.predict(X_test_label)

        combined_preds = mix_lgr*lgr_preds + (1.0 - mix_lgr)*rf_preds

        filename = raw_input("Enter name for submission file: ")
        save_results(combined_preds, filename + ".csv")
        return 0

if __name__ == '__main__':
    if sys.argv[1] == "test":
        # (trees, min_samples_split, min_samples_leaf, C, mix_lgr)
        param_sets = [
                      (100, 8, 1, 3, 0.6), #Mean AUC: 0.874386 -- the one to beat
                     ]

        for params in param_sets:
            print("trees:{} min_samples_split:{} min_samples_leaf:{} C:{} mix_lgr:{}".format(*params))
            mean_auc = run_model(True, *params)
            print "Mean AUC: %f" % (mean_auc)
    elif sys.argv[1] == "run":
        mean_auc = run_model(False, 100, 8, 1, 3, 0.6)
