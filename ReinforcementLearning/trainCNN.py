import tensorflow as tf
from tensorflow import keras
import numpy as np


if __name__ == "__main__":

    states = np.load('EpisodeLogs/sAug.npy')
    Q = np.load('EpisodeLogs/QAug.npy')

    Q = Q.reshape((Q.shape[0], 1))
    states = states.reshape((states.shape[0], 4, 4, 1))

    model = keras.models.load_model('CNN.h5')

    model.fit(states, Q, epochs=1)
    model.save('CNNoR.h5', overwrite=True)
