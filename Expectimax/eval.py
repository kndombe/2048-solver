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
    return (merging_potential + feature_array["empty_tiles"]) * feature_array["max_tile"] * 0.5

def merge(row):
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

def move(board, direction):
    '''
    direction:
        0: left
        1: down
        2: right
        3: up
    '''
    # treat all direction as left (by rotation)
    board_to_left = np.rot90(board, -direction)
    for row in range(board.shape[0]):
        core = merge(board_to_left[row])
        board_to_left[row, :len(core)] = core
        board_to_left[row, len(core):] = 0

    # rotation to the original
    return np.rot90(board_to_left, direction)

def board_tester(board, direction):
    return move(board, direction)

def eval_options(board, depth):
    #Given the board, board
    if depth == 4:
        return 0, 4

    scores = []
    #Left = 0, Down = 1, Right = 2, Up = 3
    #for each possible action:
    for i in range(4):
        new_board = board_tester(board.copy(), i)
        print(f"{depth}, {i}: {new_board}")
        #extract features of result
        f = FeatureExtractor(new_board)
        #get score
        score = calculate_score(f.getfeatures())
        #if loss:
        if score == 0:
            continue
        scores.append(score + eval_options(new_board, depth+1)[0])
    print(f"{depth}: {scores}")
    max_score = max(scores)
    return max_score, scores.index(max_score)

g = Game()
for _ in range(3):
    print(g.board)
    f = FeatureExtractor(g.board)
    print(f.getfeatures())
    g.move(eval_options(g.board, 0)[1])
