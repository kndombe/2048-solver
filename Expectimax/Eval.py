import numpy as np

def calculate_score(feature_array, weights):
    score = 0
    num_features = feature_array.shape[0]
    for i in range(num_features):
        if feature_array[i][1] != 0:
            score += weights[i]
    return score

def update_weights(result, weights):
    pass

def eval_options(board, feature_array, depth):
    #Given the board, board
    if depth == 4:
        return 0, 4
    #Up = 0, Right = 1, Down = 2, Left = 3
    #for each possible action:
        #extract features of result
        #get score
        #if loss:
            #continue
        #scores.append(score + eval_options(new board, new feature_array, depth+1))

    #return max(score), action index

    pass