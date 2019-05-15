from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

import pandas
import numpy as np

df_train = pandas.read_csv('C:/Users/Kanverse/Documents/train1_scaled.csv', header=0)
df_val = pandas.read_csv('C:/Users/Kanverse/Documents/validation1_scaled.csv', header=0)

arr_train = df_train.values
arr_val = df_val.values

def x_test():
    test = arr_val[:, 0:4]
    x_t = test
    return x_t


def x_train():
    train = arr_train[:, 0:4]
    x_tr = train
    X = x_tr
    return X


def y_test():
    y_t = arr_val[:, -1]
    return y_t


def y_train():
    y_tr = arr_train[:, -1]
    y_shaped = y_tr.reshape(-1, 1)
    y_ravel = y_shaped.ravel()
    y = y_ravel
    return y


def train():
    X = x_train()
    y = y_train()
    clf = MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(18,4), learning_rate='adaptive',
       learning_rate_init=0.0005, max_iter=200000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
    clf.fit(X, y)
    joblib.dump(clf, 'C:/Users/Kanverse/Documents/enrol_pred.pkl')

    results = clf.predict(X)

    output_predictions_file = 'C:/Users/Kanverse/Documents/prdictions.txt'
    np.savetxt(output_predictions_file, results, fmt='%.6f')
    return clf


def train_error():
    x = x_train()
    y = y_train()
    clf = train()
    score_train = clf.score(x, y)
    score_train *= 100
    train_error = round((100 - score_train), 1)
    score_error = str(train_error)
    print('Training Error: ' + score_error + '%')


def test_accuracy():
    clf = train()
    xt = x_test()
    yt = y_test()
    score_test = clf.score(xt, yt)
    score_test *= 100
    score_test = round(score_test, 1)
    score_test = str(score_test)
    print('Test Accuracy: ' + score_test + '%')


def main():
    train()
    train_error()
    test_accuracy()


main()
