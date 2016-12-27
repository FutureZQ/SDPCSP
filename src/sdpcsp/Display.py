"""
 a module for displaying the result of classification
"""
import numpy as np

def caculateaccuracyrate(pre_labels, true_labels):
    """
    a module for displaying the result of classification
    """
    print('The accuracy is:' +
          str(np.average(pre_labels == true_labels) * 100) + r'%')
