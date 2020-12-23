import numpy as np


# open text file and read in data as `text`
with open('data/anna.txt', 'r') as f:
    text = f.read()


def one_hot_encode(arr, n_labels):

    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot
