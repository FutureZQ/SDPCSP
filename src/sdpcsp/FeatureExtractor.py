"""
a module contain function to extract  features of data
"""
import numpy as np
# class FilterNumberError(Exception):
#     """the error happen when user set the wrong number of filters (odd)"""
# message= ''
# def __init__(self, msg):
#     Exception.__init__(self)
#     message=msg

def csptrain(data, labels, filter_number=2):
    """
    train the CSP , output the filter W and the features which is used to train classfier

    :parameter 1: the train_data
    :parameter 2: the train_labels
    :parameter 3: the the spatial filter used
    :returen 1: fiters,the spatial filter created by csp
    :returen 2: features,the features usd used to train classfier
    """
    cov_matrix_list = __getdiffclasscov(data, labels)
    filters = __getcspfilter(cov_matrix_list, filter_number)
    features = featureextract(filters, data)
    return filters, features


def featureextract(filters, data):
    """
    Extract feature with certain spatial filters

    :parameter 1: fiters,the spatial filter created by csp
    :parameter 2: data to be extracted features
    :returen : features,the features usd used to train classfier
    """
    data_spatial_filted = __spatialfilt(filters, data)
    features = __caculatedatapower(data_spatial_filted)
    return features

"""
Following  is private function
"""


def __caculatedatapower(data_spatial_filted):
    feature_number = np.shape(data_spatial_filted)[2]
    feature_dimension = np.shape(data_spatial_filted)[1]
    features = np.zeros((feature_number, feature_dimension))

    for i in range(0, feature_number):
        temp_data = data_spatial_filted[:, :, i]
        for j in range(0, feature_dimension):
            features[i, j] = np.sum(temp_data[:, j] * temp_data[:, j])

    features_sum = np.sum(features * features, axis=1)
    features_sum = np.column_stack((features_sum, features_sum))
    features = features / features_sum
    features = np.log10(features)
    return features


def __spatialfilt(filters, data):
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


def __getcspfilter(cov_matrix_list, filter_number=2):
    # if filter_number % 2 != 0:
    #     raise FilterNumberError('filter number must be Even')
    b_matrix = np.matrix(cov_matrix_list[0] + cov_matrix_list[1])
    a_matrix = np.matrix(cov_matrix_list[0])
    hy_matrix = np.dot(b_matrix.I, a_matrix)
    weight, vector = np.linalg.eig(hy_matrix)
    index = np.argsort(weight)[::-1]
    vector = vector[:, index]
    vector_row_number = np.shape(vector)[0]
    filters = np.zeros(shape=[vector_row_number, filter_number])
    for i in range(0, filter_number, 2):
        filters[:, i] = np.squeeze(vector[:, i])
        filters[:, i + 1] = np.squeeze(vector[:, -1 - i])
    return filters


def __getdiffclasscov(data, labels):
    unique_labels = __finduniquelabel(labels)
    cov_matrix_list = []
    for unique_label in unique_labels:
        index = np.where(labels == unique_label)
        index = __tupletolist(index)
        certarin_class_data_cov = __getcov(data[:, :, index])
        cov_matrix_list.append(certarin_class_data_cov)
    return cov_matrix_list


def __tupletolist(index):
    index = np.array(index)
    index_finally = []
    for value in index[0, :]:
        index_finally.append(int(value))
    return index_finally


def __finduniquelabel(labels):
    unique_label = []
    for label in labels:
        if label not in unique_label:
            unique_label.append(label)
    return np.array(unique_label)


def __getcov(data):
    channel_number = np.shape(data)[1]
    sample_number = np.shape(data)[-1]
    cov_sum = np.zeros((channel_number, channel_number))
    for index in range(0, sample_number):
        sample = data[:, :, index]
        sample_cov = np.dot(sample.T, sample) / sample_number
        cov_sum = cov_sum + sample_cov
    cov_sum = cov_sum / np.trace(cov_sum)
    return cov_sum