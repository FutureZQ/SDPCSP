"""
    A module to preprocess EEG signal (filt)
"""
import scipy.signal as signal


def preprocessdata(data_raw, labels, pass_band, sample_freqency=512):
    """
    A method to find preprocess data which used for tesing

    :param param1: train data or test data (3d matrix)
    :param param2: labels (it's no use now,reserve for future)
    :param param3: pass_band, the format is [low,high]
    :param param4: sample_freqency
    :returns1: data_filted, a data filted by bandpass filter
    :returns2: labels(the input one,no change)

    Note: the format of data_raw is a 3d matrix ,such that:

    data_raw[0]:
     1  2   3   4   5   6   7 .......... Channel Number
    t1
    t2
    t3
    ...
    ...
    sample_point

    data_raw[1]:
    1  2   3   4   5   6   7 .......... Channel Number
    t1
    t2
    t3
    ...
    ...
    sample_point

    .....

    data_raw[-1]:

    """
    data_filted = __eegfilt(data_raw, pass_band, sample_freqency)
    return data_filted, labels

"""
Following  is private function
"""


def __eegfilt(data, pass_band, sample_freqency):
    pass_band = [value / sample_freqency * 2 for value in pass_band]
    zero_point, pole = signal.butter(3, pass_band, 'band')
    # axis=0 is mean that each row of data is a time point
    data_filted = signal.filtfilt(zero_point, pole, data, axis=0)
    return data_filted
