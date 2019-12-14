import numpy as np
import copy
from util import *

moveDict = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None, log=False, logFileName='TransitionLog', diversify=False):
        self.game = game
        self.display = display
        self.log = log
        if self.log:
            self.logger = episodeLogger(logFileName, diversify)

    def __closeLogger__(self):
        if self.log:
            self.logger.__close__()

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
        if self.log:
            self.__closeLogger__()

    def playAndLog(self, max_iter=np.inf, verbose=False):
        n_iter = 1
        while (n_iter <= max_iter) and (not self.game.end):
            s = self.game.board
            preTileCount = np.count_nonzero(s)

            direction = self.step()
            self.game.move(direction)

            r = preTileCount - np.count_nonzero(self.game.board) + 1

            if self.game.end or n_iter == max_iter:
                r = -1

            self.logger.logTransition(s, direction, r)

            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
        self.__closeLogger__()
        return n_iter - 1

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class CNNAgent(Agent):
    def __init__(self, game, model, epislon=0.2, display=None, log=False, logFileName='TransitionLog', diversify=False):
        self.model = model
        self.epislon = epislon
        super().__init__(game, display=display, log=log,
                         logFileName=logFileName, diversify=diversify)

    def play(self, max_iter=np.inf, debug=False):
        # n_iter = 0
        while not self.game.end:
            direction = self.step(debug)
            self.game.move(direction)
            # n_iter += 1

    def playAndRecord(self):
        states = []
        actions = []
        interState = []
        merges = []
        scores = []
        turn = 0
        while not self.game.end:
            s = self.game.board
            preTileCount = np.count_nonzero(s)
            direction = self.step()
            self.game.moveWithoutSpawn(direction)

            mer = preTileCount - np.count_nonzero(self.game.board) + 1
            states.append(s.copy())
            actions.append(direction)
            interState.append(self.game.board)
            merges.append(mer)
            scores.append(self.game.score)

            self.game._maybe_new_entry()

            turn += 1
        return (np.array(states), np.array(actions), np.array(interState), np.array(merges), np.array(scores), turn)

    def step(self, debug=False):
        lM, lS = self.game.legalSuccessors()

        numMoves = len(lM)
        if numMoves == 0:
            return 0
        if numMoves == 1:
            return lM[0]

        if np.random.random() < self.epislon:
            return lM[np.random.randint(0, numMoves)]

        successors = np.array(lS)
        successors[np.where(successors == 0)] = 1
        successors = np.log2(successors)
        Q = self.model.predict(successors)

        if debug:
            print(self.game.board)
            for i in range(len(lM)):
                print(lM[i])
                print(lS[i].reshape((4, 4)))
            print(Q)
            print(moveDict[lM[np.argmax(Q)]])
            if len(input('next')) > 0:
                exit(0)
            print('')

        return lM[np.argmax(Q)]


class FCAgent(Agent):

    def __init__(self, game, model, epislon=0.07, display=None, logFileName='TransitionLog', diversify=True):
        self.model = model
        self.epislon = epislon
        super().__init__(game, display, logFileName, diversify)

    def play(self, max_iter=np.inf, debug=False):
        # n_iter = 0
        while not self.game.end:
            direction = self.step(debug)
            self.game.move(direction)
            # n_iter += 1

    def playAndRecord(self):
        states = []
        actions = []
        interState = []
        merges = []
        scores = []
        turn = 0
        while not self.game.end:
            s = self.game.board
            preTileCount = np.count_nonzero(s)
            direction = self.step()
            self.game.moveWithoutSpawn(direction)

            mer = preTileCount - np.count_nonzero(self.game.board) + 1
            states.append(s.copy())
            actions.append(direction)
            interState.append(self.game.board)
            merges.append(mer)
            scores.append(self.game.score)

            self.game._maybe_new_entry()

            turn += 1
        return (np.array(states), np.array(actions), np.array(interState), np.array(merges), np.array(scores), turn)

    def step(self, debug=False):

        s = self.game.board.reshape(1, 16)

        # uncomment if use logged representation
        # s[np.where(s == 0)] = 1
        # s = np.log2(s)

        legalMoves = self.game.legalMoves()
        numMoves = len(legalMoves)
        if numMoves == 0:
            return 0
        if numMoves == 1:
            return legalMoves[0]

        if np.random.random() < self.epislon:
            return legalMoves[np.random.randint(0, numMoves)]

        Q = self.model.predict(
            [np.repeat(s, numMoves, axis=0), np.array(legalMoves, dtype=float)])
        if debug:
            print(self.game.board)
            print(legalMoves)
            print(Q)
            print(moveDict[legalMoves[np.argmax(Q)]])
            if len(input('next')) > 0:
                exit(0)
            print('')

        return legalMoves[np.argmax(Q)]


class BaseLineAgent(Agent):

    def play(self, debug=False):
        super().play()

    def step(self):
        legalMoves = self.game.customOrderLM()
        numMoves = len(legalMoves)
        if numMoves == 0:
            return 0
        return legalMoves[0]


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None, logFileName='TransitionLog', diversify=True):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display, logFileName, diversify)
        from expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction
