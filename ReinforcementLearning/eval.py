from agents import *
from game import *
from tensorflow import keras

import numpy as np

NUM_EPISODES = 10

MODEL_NAME = 'CNN.h5'
model = keras.models.load_model(MODEL_NAME)

if __name__ == "__main__":
    scores = np.zeros(NUM_EPISODES)
    maxTiles = np.zeros(NUM_EPISODES)
    for epis in range(NUM_EPISODES):
        # agent = DQNAgent(game, modelName='dqnComplexReward.h5')
        # agent = DQNAgent(game, modelName='epoch20.h5')
        # agent = DQNAgent(game, modelName='dqnsmallE.h5')
        # agent = DQNAgent(game, modelName='selfLT')
        # agent = FCAgent(Game(), model, epislon=0)
        # agent = CornerAgent(Game())
        agent = CNNAgent(Game(), model, epislon=0)

        agent.play(debug=False)
        # agent.play(debug=True)

        scores[epis] = agent.game.score
        maxTiles[epis] = agent.game.maxTile
        print('Episode {} finished, maxTile: {}, score: {}'.format(
            epis + 1, agent.game.maxTile, agent.game.score))
    print('Average Score: ', np.mean(scores))
    largest = np.max(maxTiles)
    print('Percentage of achieving max tile of {} is {}'.format(
        largest, sum(maxTiles == largest) / NUM_EPISODES * 100))
    secLargest = np.max(maxTiles[np.where(maxTiles < largest)])
    print('Percentage of achieving max tile of {} is {}'.format(
          secLargest, sum(maxTiles == secLargest) / NUM_EPISODES * 100))
    thiLargest = np.max(maxTiles[np.where(maxTiles < secLargest)])
    print('Percentage of achieving max tile of {} is {}'.format(
          thiLargest, sum(maxTiles == thiLargest) / NUM_EPISODES * 100))
