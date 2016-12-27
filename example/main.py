#!usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
# include sdpcsp path
FIEL_PATH = sys.path[0]
WORK_SPACE_PATH = os.path.dirname(FIEL_PATH)
sys.path.append(WORK_SPACE_PATH)
sys.path.append(WORK_SPACE_PATH + os.sep + 'src')
# from sdpcsp import *
import sdpcsp.EEGReader as EEGReader
import sdpcsp.PreProcess as PreProcess
import sdpcsp.FeatureExtractor as FeatureExtractor
import sdpcsp.Classifier as Classifier
import sdpcsp.Display as Display


def main():
    """
    Main:
        first part:
        1. read train data and test data from disk

        second part:
        1. preprocess the train data with a frequency pass band
        and reshape the data to a 3d matrix.
        2. train the feature extractor (CSP) to get spatial filter
        3. train the classifier (SVM)

        third part
        1. preprocess the test data with a frequency pass band
        and reshape the data to a 3d matrix.
        2. obtain the features of test data by spatial filter
        3. obtain the predict labels uing the classifier

        Last part:
        1 display the result of claasification
    """
    # parametter setting
    pass_band = [10, 35]
    filters_number = 2
    sample_frquency = 512

    # read train_data and test_data from certain directory
    train_data, train_labels, test_data, test_labels = EEGReader.getdata(
        'example' + os.sep + 'data', 0.9, sample_frquency)

    # preprocess train data
    train_data, train_labels = PreProcess.preprocessdata(
        train_data, train_labels, pass_band, sample_frquency)
    # for two-class situation
    train_data = train_data[:, :, train_labels != 3]
    train_labels = train_labels[train_labels != 3]
    # train features exctractor
    filters, features_train = FeatureExtractor.csptrain(
        train_data, train_labels, filters_number)
    # train classifier
    classifier = Classifier.trainclassifier(features_train, train_labels)

    # preprocess test data
    test_data, test_labels = PreProcess.preprocessdata(
        test_data, test_labels, pass_band, sample_frquency)
    # for two-class situation
    test_data = test_data[:, :, test_labels != 3]
    test_labels = test_labels[test_labels != 3]
    # extractor features
    test_features = FeatureExtractor.featureextract(filters, test_data)
    # classification
    pre_labels = Classifier.predirectclass(classifier, test_features)

    # display the result
    Display.caculateaccuracyrate(pre_labels, test_labels)


if __name__ == '__main__':
    # Entry of program: main fuction
    sys.exit(main())
