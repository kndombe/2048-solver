import tensorflow as tf
from tensorflow import keras
import numpy as np

DISCOUNT = 0.9

NUM_STEP = 150
BATCH_SIZE = 32
TRAIN_SIZE = NUM_STEP * BATCH_SIZE

if __name__ == "__main__":
    model = keras.models.load_model('FC')
    ar = np.genfromtxt('EpisodeLogs/rate2corrected_ar', delimiter=',')
    reward = ar[:, 1]
    action = ar[:, 0]
    states = np.genfromtxt('EpisodeLogs/rate2corrected_s', delimiter=',')

    numEx = states.shape[0]

    Qp = np.zeros((TRAIN_SIZE, 4))

    for cycle in range(100):
        chosen = np.random.choice(np.linspace(
            0, numEx - 2, num=numEx - 1).astype(int), TRAIN_SIZE, replace=False)

        sTrain = states[chosen]
        sP = states[chosen + 1]
        aTrain = action[chosen]
        rTrain = reward[chosen]
        # Self label
        print('Start labeling Q for cycle {}'.format(cycle + 1))
        for dir in range(4):
            Qp[:, dir] = model.predict(
                [sP, np.ones(TRAIN_SIZE) * dir]).flatten()
        Q = rTrain + DISCOUNT*np.max(Qp, axis=1)

        print('Cycle {} Q labelled'.format(cycle + 1))

        model.fit([sTrain, aTrain], Q, epochs=1)
    model.save('FC', overwrite=True)
