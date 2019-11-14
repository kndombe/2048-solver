import numpy as np
import random

from features import FeatureExtractor

import sys
sys.path.append('../APIs')
from game import Game

def calculate_score(feature_array):
    #the score is dependent on the number of mergeable things and the number of empty tiles
    if feature_array["empty_tiles"] == 0: #there are no empty tiles
        return 0

    vertical_merge = 0
    vertical_merge_value = 0
    horizontal_merge = 0
    horizontal_merge_value = 0
    for feature in feature_array:
        if "has_merge_col" in feature:
            vertical_merge += 1
            vertical_merge_value += feature_array[feature]
        if "has_merge_row" in feature:
            horizontal_merge += 1
            horizontal_merge_value += feature_array[feature]

    merging_potential = max((vertical_merge * vertical_merge_value), (horizontal_merge * horizontal_merge_value))
    return merging_potential + feature_array["empty_tiles"]

def board_tester(board, action):
    pass

def eval_options(board, depth):
    #Given the board, board
    if depth == 4:
        return 0, 4

    scores = []
    #Up = 0, Right = 1, Down = 2, Left = 3
    #for each possible action:
    for i in range(4):
        new_board = board_tester(board, i)
        #extract features of result
        f = FeatureExtractor(new_board)
        #get score
        score = calculate_score(f.getfeatures())
        #if loss:
        if score == 0:
            continue
        scores.append(score + eval_options(new_board, depth+1)[0])
    max_score = max(scores)
    return max_score, scores.index(max_score)

g = Game()
for _ in range(10):
    f = FeatureExtractor(g.board)
    g.move(random.randint(0, 3))
    print(g.board)
    print(f.getfeatures())
    print(calculate_score(f.getfeatures()))
