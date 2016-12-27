"""a module to buid a classifier"""
import numpy as np
import sklearn.svm as svm

def trainclassifier(features_train, labels_train):
    """
    A method train the Classifier

    :param param1: class
    :param param2: the features used to classify
    :param param3: the labels responding to the features
    :returns: a trained classifier
    """
    classfier = svm.SVC()
    classfier.fit(features_train, labels_train)
    return classfier


def predirectclass(classfier, features_test):
    """
    A method predict the class of data

    :param param1: class
    :param param2: the classifer that has been trained
    :param param3: the features need to be classified
    :returns: a list of labels responding to the fatures
    """
    pre_lables = classfier.predict(features_test)
    return pre_lables
