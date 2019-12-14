import tensorflow as tf
from tensorflow import keras
import numpy as np

NUM_STEP = 150
BATCH_SIZE = 32
TRAIN_SIZE = NUM_STEP * BATCH_SIZE

if __name__ == "__main__":
    states = np.genfromtxt(
        'EpisodeLogs/rate2corrected_s', delimiter=',')
    actions = np.genfromtxt(
        'EpisodeLogs/rate2corrected_ar', delimiter=',')[:, 0]
    Q = np.load('Qvalues/initQ_complex.npy')

    numEx = states.shape[0]
    chosen = np.random.choice(np.linspace(
        0, numEx - 2, num=numEx - 1).astype(int), TRAIN_SIZE, replace=False)

    sTrain = states[chosen]
    aTrain = actions[chosen]
    QTrain = Q[chosen]

    model = keras.models.load_model('FC.h5')

    model.fit([sTrain, aTrain], QTrain, epochs=1)
    model.save('FC.h5', overwrite=True)
