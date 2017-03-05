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
import numpy as np

features_train = np.random.random((10, 2))
features_labels = np.zeros(10)
for i in range(0, 10):
    if (features_train[i, 0] >= features_train[i, 1]):
        features_labels[i] = 1

classifier = Classifier.trainclassifier(features_train, features_labels)
pre_labels = Classifier.predirectclass(classifier, features_train)
Display.caculateaccuracyrate(pre_labels, features_labels)
