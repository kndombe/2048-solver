import sys
sys.path.append('../APIs')
from game import Game

import eval
import time

class Agent2048():
    def __init__(self, game, wait=0.5):
        self.game = game
        self.wait = wait

    def actions(self):
        return [0,1,2,3]

    def successor(self, action):
        return eval.board_tester(self.game.board, action)

    def isEnd(self):
        return eval.eval_options(self.board,1) == 0

    def utility(self, board):
        return eval.eval_options(board,1) #TODO: set default value for depth maybe?

    def choose_action(self):
        optimal = (None, float('-inf'))
        for a in self.actions():
            succ = self.successor(a)
            value = sum([self.utility(s) for s in succ]) / len(succ)
            if value > optimal[1]:
                optimal = (a, value)
        return optimal[0]

    def simulate(self):
        print(self.game.board)
        while not self.isEnd():
            action = self.choose_action()
            print('\nChosing action {}'.format(action))
            print(self.game.board)
            self.game.move(action)
            time.sleep(self.wait)


    
