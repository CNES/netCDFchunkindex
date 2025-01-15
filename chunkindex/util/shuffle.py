#Copyright 2025 Centre National d'Etudes Spatiales
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
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
