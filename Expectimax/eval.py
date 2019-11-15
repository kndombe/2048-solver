import numpy as np
from features import FeatureExtractor
import statistics
import sys
sys.path.append('../APIs')
from game import Game

def calculate_score(feature_array):
    #the score is dependent on the number of mergeable things and the number of empty tiles
    if len(feature_array) == 2 and feature_array["empty_tiles"] == 0: #there are no empty tiles
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

def board_generator(board, direction):
    basic_board = move(board, direction)
    successors = []
    for r in range(4):
        for c in range(4):
            if basic_board[r][c] == 0:
                basic_board[r][c] = 2
                successors.append(basic_board.copy())
                basic_board[r][c] = 4
                successors.append(basic_board.copy())
                basic_board[r][c] = 0
    return successors

def eval_options(board, depth, max_depth=1):
    #Given the board, board
    if depth == max_depth:
        return 0, 4

    scores = []
    #Left = 0, Down = 1, Right = 2, Up = 3
    #for each possible action:
    for i in range(4):
        successors = board_generator(board.copy(), i)
        for new_board in successors:
            #extract features of result
            f = FeatureExtractor(new_board)
            #get score
            score = calculate_score(f.getfeatures())
            #if loss:
            if score == 0:
                continue
            scores.append(score + eval_options(new_board, depth+1)[0])
    max_score = max(scores)
    return statistics.mean(scores), scores.index(max_score)

g = Game()
for turn in range(5000):
    print(f"Turn {turn}!")
    print(g.board)
    f = FeatureExtractor(g.board)
    print(f.getfeatures())
    score = calculate_score(f.getfeatures())
    if(score == 0):
        print(f"You lost on turn {turn} with a score of {score}!")
        break
    g.move(eval_options(g.board, 0, 2)[1])
print(g.board)

#print(calculate_score({'empty_tiles': 0, 'has_merge_row_0_23': 4.0, 'has_merge_col_0_12': 8.0, 'has_merge_col_1_01': 2.0, 'has_merge_col_1_23': 32.0, 'has_merge_col_2_01': 4.0, 'has_merge_col_3_23': 4.0, 'max_tile': 32.0}))