import numpy as np


def shuffle(barray, shuffle_opt):
    """
    Apply byte-shuffle to the input binary array.

    :param barray:      input binary array
    :param shuffle_opt: number of bytes to shuffle
    :return: the byte-shuffle binary array
    """
    # Convert the input binary array to a numpy array
    array = np.frombuffer(barray, dtype='b')
    # Reshape the array to have one sample on each line and its bytes on the columns
    array = np.reshape(array, (-1, shuffle_opt))
    # Transpose the array to have one sample on each column and its bytes on the lines
    array = array.transpose()
    # Reshape the array to 1D reading line by line, i.e. all first bytes, then all second bytes, etc. 
    array = np.reshape(array, (1, -1)).squeeze().tobytes()
    return array


def unshuffle(barray, shuffle_opt):
    """
    Apply byte-unshuffle to the input binary array.

    :param barray:      input binary array
    :param shuffle_opt: number of bytes to unshuffle
    :return: the byte-unshuffle binary array
    """
    # Convert the input binary array to a numpy array
    array = np.frombuffer(barray, dtype='b')
    # Reshape the array to have one sample on each column and its bytes on the lines
    array = np.reshape(array, (shuffle_opt, -1))
    # Transpose the array to have one sample on each line and its bytes on the columns
    array = array.transpose()
    # Reshape the array to 1D reading line by line, i.e. the first sample, then the second, etc. 
    array = np.reshape(array, (1, -1)).squeeze().tobytes()
    return array
