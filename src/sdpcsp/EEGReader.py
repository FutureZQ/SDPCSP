"""A module that have some function to read EEG file"""
import os
import numpy as np


def getdata(directory, ratio, sample_freqency):
    """
    A method to get data

    :param param1: class
    :param param2: EEG File directory
    :param param3: ratio, train_data_number / test_data_number
    :returns: It will return a data and a label of numpy.array type
    :raises FileNotFoundError: not found the directory.
    """
    data, labels = __getrawdataset(directory)
    data_3d, labels_3d = __get3ddata(data, labels, sample_freqency)
    train_data, train_labels, test_data, test_labels = __dividetrainandtestdata(
        data_3d, labels_3d, ratio)
    return train_data, train_labels, test_data, test_labels


"""
Following  is private function
"""


def __getrawdataset(directory):

    file_list = __getfilelist(directory)
    data_raw = __readfiles(file_list)
    data, labels = __datadivide(data_raw)
    return data, labels


def __readfile(eeg_file_name):
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


def __readfiles(eeg_file_list):
    eeg_data = None
    for file in eeg_file_list:
        temp_data = __readfile(file)
        if eeg_data is None:
            eeg_data = temp_data
        else:
            eeg_data = np.concatenate((eeg_data, temp_data))
    return eeg_data


def __getfilelist(directory):
    all_file_list = os.listdir(directory)
    eeg_file_list = []
    for file_name in all_file_list:
        if str(file_name).endswith('.asc') and str(file_name).startswith('train'):
            eeg_file_list.append(directory + os.sep + file_name)
    return eeg_file_list


def __datadivide(data_raw):
    data = data_raw[:, 0:-1]
    labels = data_raw[:, -1]
    return data, labels


def __dividetrainandtestdata(data, labels, ratio):
    train_data_len = round(np.shape(data)[2] * ratio)

    train_data = data[:, :, 0:train_data_len]
    train_labels = labels[0:train_data_len]

    test_data = data[:, :, train_data_len:]
    test_labels = labels[train_data_len:]

    return train_data, train_labels, test_data, test_labels


def __get3ddata(data, labels, sample_freqency=512,):
    """
    cut the border of different class data, and reshape the data into
    """
    sample_number, data_struct = __getstruct(labels, sample_freqency)
    channl_number = __getchannelnumber(data)
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


def __getchannelnumber(data):
    return int(np.shape(data)[1])


def __getdatadistrubution(labels):
    """
    Get the data distrubute table
    Each row is a slice of data with different label
    """
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


def __adjustdistrubutionstruct(distrubution_martix, sample_freqency=512):
    """
     cut the data of border and
     get the number of frame(slices of data) in data
    """
    sample_number_total = 0
    delete_item = []
    cut_number = round(sample_freqency / 5)

    for index, item in enumerate(distrubution_martix):
        item[0] = item[0] + cut_number
        item[1] = item[1] - cut_number
        sample_number_per_frame = item[1] - item[0] + 1
        # consider some sample points of frame  are less then sample_frequncy
        if sample_number_per_frame <= sample_freqency:
            delete_item.append(index)
        sample_number_per_frame = sample_number_per_frame // sample_freqency
        if sample_number_per_frame < 1:
            sample_number_per_frame = 0
        sample_number_total = sample_number_total + sample_number_per_frame

    # delete the frames whose sample points are less then sample_frequncy
    np.delete(delete_item, distrubution_martix)
    return sample_number_total, distrubution_martix


def __getstruct(labels, sample_freqency=512):
    """
    get the start point, stop point and
    datalabel of each frame(slices of data) data
    """
    distrubution_martix = __getdatadistrubution(labels)
    sample_number, distrubution_martix = __adjustdistrubutionstruct(
        distrubution_martix, sample_freqency)
    return sample_number, distrubution_martix
