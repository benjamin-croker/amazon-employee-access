from __future__ import division

import numpy as np
from sklearn import (metrics, cross_validation, linear_model,
        preprocessing, ensemble)

SEED = 0  # always use a seed for randomized procedures


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


def main(trees, min_samples_split, max_features, C):
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.
    """
    lgr_model = linear_model.LogisticRegression(C=C)
    rf_model = ensemble.RandomForestRegressor(trees, n_jobs=2,
            min_samples_split=min_samples_split,
            max_features=max_features)
    # rf_model = ensemble.RandomForestRegressor(100, n_jobs=2)

    # === load data in memory === #
    print "loading data"
    y, X = load_data('train.csv')
    y_test, X_test = load_data('test.csv', use_labels=False)

    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    oneHotencoder = preprocessing.OneHotEncoder()
    oneHotencoder.fit(np.vstack((X, X_test)))

    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(np.vstack((X, X_test)))

    X_sparse = oneHotencoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test_sparse = oneHotencoder.transform(X_test)

    X_label = labelEncoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test_label = labelEncoder.transform(X_test)

    # if you want to create new features, you'll need to compute them
    # before the encoding, and append them to your dataset after

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
        mean_preds = 0.6*lgr_preds + 0.4*rf_preds

        # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, lgr_preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "Logistic Regression AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc)

        fpr, tpr, thresholds = metrics.roc_curve(y_cv, rf_preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "RF Regression AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc)

        fpr, tpr, thresholds = metrics.roc_curve(y_cv, mean_preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "Combined AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc)

        mean_auc += roc_auc


    if n !=0: print "Mean AUC: %f" % (mean_auc/n)

    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    lgr_model.fit(X_sparse, y) 
    rf_model.fit(X_label, y) 
    
    lgr_preds = lgr_model.predict_proba(X_test_sparse)[:, 1]
    rf_preds = rf_model.predict(X_test_label)
    mean_preds = 0.6*lgr_preds + 0.4*rf_preds

    filename = raw_input("Enter name for submission file: ")
    save_results(mean_preds, filename + ".csv")

if __name__ == '__main__':
    # (trees, min_samples_split, max_features, C)

    # main(trees=100, min_samples_split=2, max_features="auto", C=3) #Mean AUC: 0.871360
    main(trees=100, min_samples_split=1, max_features=None, C=3)
