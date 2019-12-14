import tensorflow as tf
from tensorflow import keras
import numpy as np

from agents import *
from game import *

NUM_EPISODE = 1000
MODEL_NAME = 'initFC'

DISCOUNT = 1.0


def label(states, reward):
    TRAIN_SIZE = reward.size
    Qp = np.zeros((TRAIN_SIZE - 1, 4))
    for dir in range(4):
        Qp[:, dir] = model.predict(
            [states[:-1], np.ones(TRAIN_SIZE - 1) * dir]).flatten()
    Q = reward.copy().astype(float)
    Q[:-1] += DISCOUNT*np.max(Qp, axis=1)
    return Q


if __name__ == "__main__":
    model = keras.models.load_model(MODEL_NAME)
    avgScore = 0
    s = np.zeros((0, 16), dtype=int)
    a = np.zeros((0, 1))
    Q = np.zeros((0, 1))
    update = 1
    for episode in range(1, NUM_EPISODE + 1):
        agent = FCAgent(Game(), model, epislon=0.2)
        states, actions, interState, merges, scores, turn = agent.playAndRecord()
        avgScore = (avgScore * (episode - 1) + agent.game.score) / episode
        print('episode {} finished, score: {}, maxTile: {}, {} turns, average score: {}'.format(
            episode, agent.game.score, agent.game.maxTile, turn, avgScore))
        np.save('records/states_{}'.format(episode), states)
        np.save('records/actions_{}'.format(episode), actions)
        np.save('records/interStates_{}'.format(episode), interState)
        np.save('records/merges_{}'.format(episode), merges)
        np.save('records/scores_{}'.format(episode), scores)

        states = states.reshape((states.shape[0], 16))
        scores[1:] -= scores[:-1]

        curQ = label(states, scores.flatten())

        s = np.vstack((s, states))
        a = np.vstack((a, actions.reshape((actions.shape[0], 1))))
        Q = np.vstack((Q, curQ.reshape((curQ.shape[0], 1))))

        if Q.size > 1000:
            model.fit([s, a], Q)
            s = np.zeros((0, 16), dtype=int)
            a = np.zeros((0, 1))
            Q = np.zeros((0, 1))
            if update % 10 == 0:
                model.save('exFC', overwrite='True')
            update += 1

        # there's memory leak issue, likely from keras
        del states
        del actions
        del interState
        del merges
        del scores
        del agent
        keras.backend.clear_session()

    model.save('exFC', overwrite='True')
