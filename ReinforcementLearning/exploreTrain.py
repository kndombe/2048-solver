import tensorflow as tf
from tensorflow import keras
import numpy as np

from agents import *
from game import *

NUM_EPISODE = 1000
MODEL_NAME = 'initFC'

DISCOUNT = 0.9


def label(states, reward):
    TRAIN_SIZE = reward.size
    Qp = np.zeros((TRAIN_SIZE - 1, 4))
    for dir in range(4):
        Qp[:, dir] = model.predict(
            [states[:-1], np.ones(TRAIN_SIZE, dtype=int) * dir]).flatten()
    Q = reward.copy()
    Q[:-1] += DISCOUNT*np.max(Qp, axis=1)
    return Q


if __name__ == "__main__":
    model = keras.models.load_model(MODEL_NAME)

    agent = FCAgent(None, model)

    for episode in range(NUM_EPISODE):
        agent.game = Game()
        states, actions, interState, merges, scores, turn = agent.playAndRecord()
        np.save('states_{}'.format(episode), states)
        np.save('actions_{}'.format(episode), actions)
        np.save('interStates_{}'.format(episode), interState)
        np.save('merges_{}'.format(episode), merges)
        np, save('scores_{}'.format(episode), scores)

        Q = label(states, merges)
        model.fit([states, actions], Q)
    model.save('FC')
