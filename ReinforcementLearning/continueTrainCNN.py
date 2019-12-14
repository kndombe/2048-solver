import tensorflow as tf
from tensorflow import keras
import numpy as np

NUM_STEP = 100
BATCH_SIZE = 32
TRAIN_SIZE = NUM_STEP * BATCH_SIZE
CYCLES = 100

DISCOUNT = 0.9

modelName = 'noRegCNNSLT.h5'


def moveWithoutSpawn(board, direction):
    board_to_left = np.rot90(board, -direction)
    for row in range(4):
        core = _merge(board_to_left[row])
        board_to_left[row, :len(core)] = core
        board_to_left[row, len(core):] = 0

    # rotation to the original
    return np.rot90(board_to_left, direction)


def _merge(row):
    '''merge the row, there may be some improvement'''
    non_zero = row[row != 0]  # remove zeros
    core = [None]
    for elem in non_zero:
        if core[-1] is None:
            core[-1] = elem
        elif core[-1] == elem:
            core[-1] = 2 * elem
            core.append(None)
        else:
            core.append(elem)
    if core[-1] is None:
        core.pop()
    return core


if __name__ == "__main__":

    model = keras.models.load_model(modelName)

    states = np.load('EpisodeLogs/sAug.npy')
    reward = np.load('EpisodeLogs/rAug.npy')
    spp = np.zeros((TRAIN_SIZE, 4, 4))
    Qp = np.zeros((TRAIN_SIZE, 4))

    numEx = states.shape[0]

    for cyc in range(CYCLES):
        chosen = np.random.choice(np.linspace(
            0, numEx - 2, num=numEx - 1).astype(int), TRAIN_SIZE, replace=False)
        s = states[chosen]
        sp = states[chosen + 1]
        r = reward[chosen]

        for dir in range(4):
            for i in range(TRAIN_SIZE):
                spp[i] = moveWithoutSpawn(sp[i], dir)
            Qp[:, dir] = model.predict(
                spp.reshape((TRAIN_SIZE, 4, 4, 1))).flatten()
        Q = r + DISCOUNT * np.max(Qp, axis=1)

        model.fit(s.reshape((TRAIN_SIZE, 4, 4, 1)),
                  Q.reshape((TRAIN_SIZE, 1)), epochs=1)
        model.save('noRegCNNSLT.h5', save_format='h5', overwrite=True)
