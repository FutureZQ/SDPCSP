"""Module docstring
A Module to deal with the EEG data
The version 0.1 alpha: The frame is finished, but there are many bug on it
"""
import os
import numpy as np
import sklearn.svm as svm
import scipy.signal as signal


class FilterNumberError(Exception):
    """the error happen when user set the wrong number of filters (odd)"""

    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg


class EEGReader():
    """A class that have some function to read EEG file"""

    def getdataset(self, directory, ratio):
        """
        A method to get data

        :param param1: class
        :param param2: EEG File directory
        :param param3: ratio, train_data_number / test_data_number
        :returns: It will return a data and a label of numpy.array type
        :raises FileNotFoundError: not found the directory.
        """
        file_list = self.__getfilelist(directory)
        data_raw = self.__readfiles(file_list)
        train_data_set, test_data_set = self.__dividetrainandtestdata(
            data_raw, ratio)
        train_data, train_labels = self.__datadivide(train_data_set)
        test_data, test_labels = self.__datadivide(test_data_set)
        return train_data, train_labels, test_data, test_labels

    def gettraindata(self, directory):
        """this function will be define in the future version """
        pass

    # def gettestdata(self, directory):
    #     pass

    # def gettestlabels(self, directory):
    #     pass

    """
    Following  is private function
    """

    def __readfile(self, eeg_file_name):
        with open(eeg_file_name, mode='r') as data_file:
            data_raw = data_file.readlines()
            row_number = len(data_raw)
            column_number = len(data_raw[0].split())
            data = np.zeros((row_number, column_number))
            current_row = 0
            for data_row in data_raw:
                data_row = data_row.split()
                data_row = [float(value) for value in data_row]
                data[current_row, :] = data_row
                current_row = current_row + 1
        return data

    def __readfiles(self, eeg_file_list):
        eeg_data = None
        for file in eeg_file_list:
            temp_data = self.__readfile(file)
            if eeg_data is None:
                eeg_data = temp_data
            else:
                eeg_data = np.concatenate((eeg_data, temp_data))

        return eeg_data

    def __getfilelist(self, directory):
        all_file_list = os.listdir(directory)
        eeg_file_list = []
        for file_name in all_file_list:
            if str(file_name).endswith('.asc') and str(file_name).startswith('train'):
                eeg_file_list.append(directory + os.sep + file_name)
        return eeg_file_list

    def __datadivide(self, data_raw):
        data = data_raw[:, 0:-1]
        labels = data_raw[:, -1]
        return data, labels

    def __dividetrainandtestdata(self, data_raw, ratio):
        train_data_len = round(len(data_raw) * ratio)
        train_data_set = data_raw[0:train_data_len, :]
        test_data_set = data_raw[train_data_len:, :]
        return train_data_set, test_data_set


class PreProcess():
    """
    A class to preprocess EEG signal
    """

    def preprocesstraindata(self, data_raw, labels_raw, pass_band, sample_freqency=512):
        """
        A method to find preprocess data which used for tesing

        :param param1: class
        :param param2: data read from *.asc
        :param param3: the freqency pass band
        :param param4: sample_freqency
        :returns1: train_data, a data for training filted by bandpass filter
        :returns2: train_labels, a list of label responding to the train_data
        """
        train_data, train_labels = self.__processdata(
            data_raw, labels_raw, pass_band, sample_freqency, True)
        return train_data, train_labels

    def preprocesstestdata(self, data_raw, labels_raw, pass_band, sample_freqency=512):
        """
        A method to find preprocess data which used for training

        :param param1: class
        :param param2: data read from *.asc
        :param param3: the freqency pass band
        :param param4: sample_freqency
        :returns1: test_data, a data used for testing or using filted by bandpass filter
        :returns2: test_labels, a list of label responding to the test_data
        """
        test_data, test_labels = self.__processdata(
            data_raw, labels_raw, pass_band, sample_freqency, False)
        return test_data, test_labels

    """
    Following  is private function
    """

    def __processdata(self, data, labels, pass_band, sample_freqency, is_train=True):

        if is_train:
            data_3d, labels_3d = self.__get3dtraindata(
                data, labels, sample_freqency)
        else:
            data_3d, labels_3d = self.__get3dtestdata(
                data, labels, sample_freqency)
        data_filted = self.__eegfilt(data_3d, pass_band, sample_freqency)
        return data_filted, labels_3d

    def __eegfilt(self, data, pass_band, sample_freqency):
        pass_band = [value / sample_freqency * 2 for value in pass_band]
        zero_point, pole = signal.butter(3, pass_band, 'band')
        data_filted = signal.filtfilt(zero_point, pole, data, axis=0)
        return data_filted

    def __get3dtestdata(self, data, labels, sample_freqency=512):
        data_len = len(data)
        sample_number = data_len // sample_freqency
        channl_number = self.__getchannelnumber(data)
        data_3d = np.zeros(
            (sample_freqency, channl_number, int(sample_number)))
        labels_3d = np.zeros((int(sample_number),))
        for index in range(0, sample_number):
            the_range = range(index * sample_freqency,
                              (index + 1) * sample_freqency)
            data_3d[:, :, index] = data[the_range, :]
            labels_3d[index] = labels[index * sample_freqency]
        return data_3d, labels_3d

    def __get3dtraindata(self, data, labels, sample_freqency=512):
        sample_number, data_struct = self.__getstruct(labels, sample_freqency)
        channl_number = self.__getchannelnumber(data)
        labels_3d = np.zeros((int(sample_number),), dtype=int)
        data_3d = np.zeros(
            (sample_freqency, channl_number, int(sample_number)))
        point = 0

        for item in data_struct:
            start_point = int(item[0])
            interval = int(sample_freqency)
            stop_point = int(item[1]) - interval + 1
            for index in range(start_point, stop_point, interval):
                data_3d[:, :, point] = data[
                    index:index + sample_freqency, :]
                labels_3d[point] = int(item[2])
                point = point + 1
        return data_3d, labels_3d

    def __getchannelnumber(self, data):
        return int(np.shape(data)[1])

    def __getdatadistrubution(self, labels):
        start_point = 0
        stop_point = 0
        current_value = labels[0]
        distrubution_martix = None
        for index, label in enumerate(labels):
            if current_value != label:
                stop_point = index - 1
                distrubution_row = np.array(
                    [start_point, stop_point, current_value])
                if distrubution_martix is None:
                    distrubution_martix = distrubution_row
                else:
                     # this method is slow ,consider to change
                    distrubution_martix = np.vstack(
                        (distrubution_martix, distrubution_row))
                start_point = index
                current_value = label
        return distrubution_martix

    def __adjustdistrubutionstruct(self, distrubution_martix, sample_freqency=512):

        sample_number_total = 0
        delete_item = []
        cut_number = round(sample_freqency / 5)

        for index, item in enumerate(distrubution_martix):
            item[0] = item[0] + cut_number
            item[1] = item[1] - cut_number
            sample_number_per_frame = item[1] - item[0] + 1
            if sample_number_per_frame <= sample_freqency:
                delete_item.append(index)
            sample_number_per_frame = sample_number_per_frame // sample_freqency
            if sample_number_per_frame < 1:
                sample_number_per_frame = 0
            sample_number_total = sample_number_total + sample_number_per_frame

        np.delete(delete_item, distrubution_martix)
        return sample_number_total, distrubution_martix

    def __getstruct(self, labels, sample_freqency=512):

        distrubution_martix = self.__getdatadistrubution(labels)
        sample_number, distrubution_martix = self.__adjustdistrubutionstruct(
            distrubution_martix, sample_freqency)
        return sample_number, distrubution_martix


class FeatureExtractor():
    """
    Extract the feature of data
    """

    def csptrain(self, data, labels, filter_number=2):
        """
        train the CSP , output the filter W and the features which is used to train classfier

        :parameter 1: the train_data
        :parameter 2: the train_labels
        :parameter 3: the the spatial filter used
        :returen 1: fiters,the spatial filter created by csp
        :returen 2: features,the features usd used to train classfier
        """
        cov_matrix_list = self.__getdiffclasscov(data, labels)
        try:
            filters = self.__getcspfilter(cov_matrix_list, filter_number)
        except FilterNumberError as fne:
            print(fne.msg)
        features = self.featureextract(filters, data)
        return filters, features

    def featureextract(self, filters, data):
        """
        Extract feature with certain spatial filters

        :parameter 1: fiters,the spatial filter created by csp
        :parameter 2: data to be extracted features
        :returen : features,the features usd used to train classfier
        """
        data_spatial_filted = self.__spatialfilt(filters, data)
        features = self.__caculatedatapower(data_spatial_filted)
        return features

    """
    Following  is private function
    """

    def __caculatedatapower(self, data_spatial_filted):
        feature_number = np.shape(data_spatial_filted)[2]
        data_len = np.shape(data_spatial_filted)[0]
        feature_dimension = np.shape(data_spatial_filted)[1]
        features = np.zeros((feature_number, feature_dimension))
        for i in range(0, feature_number):
            temp_data = data_spatial_filted[:, :, i]
            for j in range(0, feature_dimension):
                features[i, j] = np.sum(temp_data[:, j] * temp_data[:, j])
        features = np.log10(features / data_len)
        return features

    def __spatialfilt(self, filters, data):
        filter_number = np.shape(filters)[1]
        frame_data_len = np.shape(data)[0]
        sample_number = np.shape(data)[2]
        data_spatial_filted = np.zeros(
            (frame_data_len, filter_number, sample_number))
        for index in range(0, sample_number):
            data_frame = data[:, :, index].T
            data_spatial_filted[:, :, index] = (
                np.dot(filters.T, data_frame)).T
        return data_spatial_filted

    def __getcspfilter(self, cov_matrix_list, filter_number=2):
        if filter_number % 2 != 0:
            raise FilterNumberError('filter number must be Even')
        b_matrix = np.matrix(cov_matrix_list[0] + cov_matrix_list[1])
        a_matrix = np.matrix(cov_matrix_list[0])
        hy_matrix = np.dot(b_matrix.I, a_matrix)
        weight, vector = np.linalg.eig(hy_matrix)
        index = np.argsort(weight)
        vector = vector[:, index]
        vector_row_number = np.shape(vector)[0]
        filters = np.zeros(shape=[vector_row_number, filter_number])
        for i in range(0, filter_number, 2):
            filters[:, i] = np.squeeze(vector[:, i])
            filters[:, i + 1] = np.squeeze(vector[:, -1 - i])
        return filters

    def __getdiffclasscov(self, data, labels):
        unique_labels = self.__finduniquelabel(labels)
        cov_matrix_list = []
        for unique_label in unique_labels:
            index = np.where(labels == unique_label)
            index = self.__tupletolist(index)
            certarin_class_data_cov = self.__getcov(data[:, :, index])
            cov_matrix_list.append(certarin_class_data_cov)
        return cov_matrix_list

    def __tupletolist(self, index):
        index = np.array(index)
        index_finally = []
        for value in index[0, :]:
            index_finally.append(int(value))
        return index_finally

    def __finduniquelabel(self, labels):
        unique_label = []
        for label in labels:
            if label not in unique_label:
                unique_label.append(label)
        return np.array(unique_label)

    def __getcov(self, data):
        channel_number = np.shape(data)[1]
        sample_number = np.shape(data)[-1]
        cov_sum = np.zeros((channel_number, channel_number))
        for index in range(0, sample_number):
            sample = data[:, :, index]
            sample_cov = np.dot(sample.T, sample) / sample_number
            cov_sum = cov_sum + sample_cov
        cov_sum = cov_sum / np.trace(cov_sum)
        return cov_sum


class Classifier():
    """a class to buid a classifier"""

    def trainclassifier(self, features_train, labels_train):
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

    def predirectclass(self, classfier, features_test):
        """
        A method predict the class of data

        :param param1: class
        :param param2: the classifer that has been trained
        :param param3: the features need to be classified
        :returns: a list of labels responding to the fatures
        """
        pre_lables = classfier.predict(features_test)
        return pre_lables


class Screen():
    """Display the result of classification"""

    def caculateaccuracyrate(self, pre_labels, true_labels):
        """Display the accuaracy of classification"""
        print('The accuracy is:' +
              str(np.average(pre_labels == true_labels) * 100) + r'%')
