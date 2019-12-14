from agents import *
from game import *
import time

NUM_EPISODES = 1

if __name__ == "__main__":
    for epis in range(NUM_EPISODES):
        game = Game()
        agent = ExpectiMaxAgent(
            game, logFileName='EpisodeLogs/season3', diversify=False)
        startTime = time.time()
        numSteps = agent.playAndLog()
        episodeTime = time.time() - startTime
        print('Episode {} collected, max tile: {}, had {} steps, took {} seconds, avg {} s/step'.format(
            epis + 1, game.score, numSteps, episodeTime, episodeTime / numSteps))
