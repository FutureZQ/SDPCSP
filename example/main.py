#!usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
FIEL_PATH = sys.path[0]
WORK_SPACE_PATH = os.path.dirname(FIEL_PATH)
sys.path.append(WORK_SPACE_PATH)
sys.path.append(WORK_SPACE_PATH + os.sep + 'src')
import sdpcsp

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

    eeg_reader = sdpcsp.EEGReader()
    eeg_preprossor = sdpcsp.PreProcess()
    eeg_extractor = sdpcsp.FeatureExtractor()
    eeg_classifier = sdpcsp.Classifier()
    eeg_screen = sdpcsp.Screen()

    # read train_data and test_data from certain directory
    train_data, train_labels, test_data, test_labels = eeg_reader.getdataset(
        'example' + os.sep + 'data', 0.9)

    # preprocess train data
    train_data, train_labels = eeg_preprossor.preprocesstraindata(
        train_data, train_labels, [9, 30], 512)
    # for two-class situation
    train_data = train_data[:, :, train_labels != 2]
    train_labels = train_labels[train_labels != 2]
    # train features exctractor
    filters, features_train = eeg_extractor.csptrain(
        train_data, train_labels, 2)
    # train classifier
    classifier = eeg_classifier.trainclassifier(features_train, train_labels)

    # preprocess test data
    test_data, test_labels = eeg_preprossor.preprocesstestdata(
        test_data, test_labels, [9, 30], 512)
    # for two-class situation
    test_data = test_data[:, :, test_labels != 2]
    test_labels = test_labels[test_labels != 2]
    # extractor features
    test_features = eeg_extractor.featureextract(filters, test_data)
    # classification
    pre_labels = eeg_classifier.predirectclass(classifier, test_features)

    # display the result
    eeg_screen.caculateaccuracyrate(pre_labels, test_labels)


if __name__ == '__main__':
    # Entry of program: main fuction
    sys.exit(main())
