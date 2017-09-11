#!/usr/bin/env python
# -*- coding:utf-8 -*-


import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import scipy
from scipy import ndimage


def load_dataset():
    train_dataset = h5py.File('dataset/test.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('dataset/train.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def init_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("cost after %i:%f" % (i, cost))

    params = {"w": w,
              "b": b
              }
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        if A[0][i] > 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
    assert Y_prediction.shape == (1, m)
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.8, print_cost=False):
    w, b = init_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]

    Y_train_prediction = predict(w, b, X_train)
    Y_test_prediction = predict(w, b, X_test)
    print("train accuracy:{}%".format(100 - np.mean(np.abs(Y_train - Y_train_prediction)) * 100))
    print("test accuracy:{}%".format(100 - np.mean(np.abs(Y_test - Y_test_prediction)) * 100))

    return {
        "cost": costs,
        "Y_test_prediction": Y_test_prediction,
        "Y_train_prediction": Y_train_prediction,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}


if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]

    num_px = train_set_x_orig.shape[1]

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=True)
    index = 14
    print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[int(
        d["Y_test_prediction"][0, index])].decode("utf-8") + "\" picture.")

    plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    plt.show()