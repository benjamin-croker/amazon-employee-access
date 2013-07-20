from __future__ import division

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn import (metrics, cross_validation, linear_model,
                     preprocessing, ensemble, svm)

SEED = 111  # always use a seed for randomized procedures


def roc_eval(actual_values, predicted_values, fig=None, label=None):
    """
    Returns the ROC area under curve
    :param predicted_values:
    :param actual_values:
    :param fig: the matplotlib axis to plot on, if none is give, a new
        figure is created
    :param label: the label to use for the ROC line
    """
    fpr, tpr, thresholds = metrics.roc_curve(actual_values, predicted_values)
    roc_auc = metrics.auc(fpr, tpr)

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(fpr, tpr, label=label)
    ax.set_xlabel("False +ve Rate")
    ax.set_ylabel("True +ve Rate")

    return roc_auc


def predictions_dist(actual_values, predicted_values, fig=None):
    posProbs = predicted_values[actual_values == 1]
    negProbs = predicted_values[actual_values == 0]

    if fig is None:
        fig = plt.figure()
    axPos = fig.add_subplot(211)
    axPos.hist(posProbs, 20)
    axPos.set_xlabel("Probs for +ve datapoints")

    axNeg = fig.add_subplot(212)
    axNeg.hist(negProbs, 20)
    axNeg.set_xlabel("Probs for -ve datapoints")


def load_data(filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s. 
    """

    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open(os.path.join("data", filename)),
                      delimiter=',', usecols=range(1, 9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open(os.path.join("data", filename)),
                            delimiter=',', usecols=[0], skiprows=1)
        return data, labels
    else:
        return data


def encode_data(X_train, X_test):
    """
    Takes the labels and data and returns label and one-hot encoded versions
    of the data
    """
    oneHotencoder = preprocessing.OneHotEncoder()
    oneHotencoder.fit(np.vstack((X_train, X_test)))

    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(np.vstack((X_train, X_test)))

    X_onehot = oneHotencoder.transform(X_train)  # Returns a sparse matrix (see numpy.sparse)
    X_test_onehot = oneHotencoder.transform(X_test)

    X_label = labelEncoder.transform(X_train)
    X_test_label = labelEncoder.transform(X_test)

    return X_onehot, X_test_onehot, X_label, X_test_label


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


def run_model(test, lgr_args, rf_args, svm_args, mix, n=10):
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
    X, y = load_data("train.csv")
    X_test = load_data("test.csv", use_labels=False)

    X_onehot, X_test_onehot, X_label, X_test_label = encode_data(X, X_test)

    # normalise the mix ratios, then mix the predictor outcomes
    mix = np.array(mix, dtype="float64")
    mix /= mix.sum()

    if test:
        # === training & metrics === #
        mean_auc = 0.0
        for i in range(n):
            # for each iteration, randomly hold out 20% of the data as CV set
            X_train_label, X_cv_label, y_train, y_cv = cross_validation.train_test_split(
                X_label, y, test_size=.20, random_state=i * SEED)

            X_train_onehot, X_cv_onehot, y_train, y_cv = cross_validation.train_test_split(
                X_onehot, y, test_size=.20, random_state=i * SEED)

            # train model and make predictions
            print("Training LGR model")
            lgr_model.fit(X_train_onehot, y_train)
            print("Training RF model")
            rf_model.fit(X_train_label, y_train)
            print("Training SVM model")
            svm_model.fit(X_train_onehot, y_train)

            lgr_preds = lgr_model.predict_proba(X_cv_onehot)[:, 1]
            rf_preds = rf_model.predict(X_cv_label)
            svm_preds = svm_model.predict_proba(X_cv_onehot)[:, 1]

            combined_preds = mix[0] * lgr_preds + mix[1] * rf_preds + mix[2] * svm_preds

            # compute AUC metric for this CV fold
            roc_fig = plt.figure()

            dist_fig = plt.figure()
            predictions_dist(y_cv, lgr_preds, fig=dist_fig)
            dist_fig.savefig(os.path.join("plots", "lgr_dist_fold_{0}.png".format(i + 1)))
            roc_auc = roc_eval(y_cv, lgr_preds, fig=roc_fig, label="LGR")
            print("Logistic Regression AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc))

            dist_fig = plt.figure()
            predictions_dist(y_cv, rf_preds, fig=dist_fig)
            dist_fig.savefig(os.path.join("plots", "rf_dist_fold_{0}.png".format(i + 1)))
            roc_auc = roc_eval(y_cv, rf_preds, fig=roc_fig, label="RF")
            print("RF Regression AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc))

            dist_fig = plt.figure()
            predictions_dist(y_cv, svm_preds, fig=dist_fig)
            dist_fig.savefig(os.path.join("plots", "svm_dist_fold_{0}.png".format(i + 1)))
            roc_auc = roc_eval(y_cv, svm_preds, fig=roc_fig, label="SVM")
            print("SVM Regression AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc))

            dist_fig = plt.figure()
            predictions_dist(y_cv, combined_preds, fig=dist_fig)
            dist_fig.savefig(os.path.join("plots", "combined_dist_fold_{0}.png".format(i + 1)))
            roc_auc = roc_eval(y_cv, combined_preds, fig=roc_fig, label="Combined")
            print("Combined AUC (fold {0}/{1}): {2}".format(i + 1, n, roc_auc))

            # roc_fig.title("ROCs for fold {0}/{1}".format(i + 1, n))
            plt.figure(roc_fig.number)
            plt.legend()
            roc_fig.savefig(os.path.join("plots", "roc_fold_{0}.png".format(i + 1)))

            mean_auc += roc_auc
        return mean_auc / n

    else:
    # === Predictions === #
        # When making predictions, retrain the model on the whole training set
        print("Training LGR model")
        lgr_model.fit(X_onehot, y)
        print("Training RF model")
        rf_model.fit(X_label, y)
        print("Training SVM model")
        svm_model.fit(X_onehot, y)

        lgr_preds = lgr_model.predict_proba(X_test_onehot)[:, 1]
        rf_preds = rf_model.predict(X_test_label)
        svm_preds = svm_model.predict_proba(X_test_onehot)[:, 1]

        combined_preds = mix[0] * lgr_preds + mix[1] * rf_preds + mix[2] * svm_preds

        filename = raw_input("Enter name for submission file: ")
        save_results(combined_preds, filename + ".csv")
        return 0


if __name__ == '__main__':
    # (lgr_args, rf_args, svm_args, mix)
    params_best = ({"C": 3},
                   {"n_estimators": 100, "min_samples_split": 8, "min_samples_leaf": 1},
                   {"C": 1, "gamma": 0.1, "cache_size": 1000},
                   [9.0, 6.0, 6.0])
    param_sets = [({"C": 3},
                   {"n_estimators": 100, "min_samples_split": 8, "min_samples_leaf": 1},
                   {"C": 1, "gamma": 0.1, "cache_size": 1000},
                   [9.0, 6.0, 6.0]), # Mean AUC: 0.870934
                  ({"C": 3},
                   {"n_estimators": 100, "min_samples_split": 8, "min_samples_leaf": 1},
                   {"C": 1, "gamma": 0.1, "cache_size": 1000},
                   [9.0, 6.0, 3.0]), # Mean AUC: 0.870739
                  ({"C": 3},
                   {"n_estimators": 100, "min_samples_split": 8, "min_samples_leaf": 1},
                   {"C": 1, "gamma": 0.1, "cache_size": 1000},
                   [6.0, 4.0, 0.0]), # Mean AUC: 0.869819
                  ({"C": 3},
                   {"n_estimators": 100, "min_samples_split": 8, "min_samples_leaf": 1},
                   {"C": 1, "gamma": 0.1, "cache_size": 1000},
                   [9.0, 3.0, 6.0]), # Mean AUC: 0.869334
                  ({"C": 3},
                   {"n_estimators": 100, "min_samples_split": 8, "min_samples_leaf": 1},
                   {"C": 1, "gamma": 0.1, "cache_size": 1000},
                   [6.0, 0.0, 4.0])  # Mean AUC: 0.860840
    ]

    if sys.argv[1] == "evaluate":
        for params in param_sets:
            print("lgr_args:{} rf_args:{} svm_args:{} mix:{}".format(*params))
            mean_auc = run_model(True, *params, n=10)
            print("Mean AUC: %f" % mean_auc)

    elif sys.argv[1] == "submission":
        params = params_best
        print("lgr_args:{} rf_args:{} svm_args:{} mix:{}".format(*params))
        mean_auc = run_model(False, *params)

    elif sys.argv[1] == "analyse":
        params = params_best
        print("lgr_args:{} rf_args:{} svm_args:{} mix:{}".format(*params))
        mean_auc = run_model(True, *params, n=1)
        print("Mean AUC: %f" % mean_auc)

    else:
        print("Unknown command: {0}".format(sys.argv[1]))