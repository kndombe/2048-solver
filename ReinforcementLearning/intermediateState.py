import numpy as np


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
    for row in range(4):
        core = _merge(board_to_left[row])
        board_to_left[row, :len(core)] = core
        board_to_left[row, len(core):] = 0

    # rotation to the original
    return np.rot90(board_to_left, direction)


def _merge(row):
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


if __name__ == "__main__":
    s = np.genfromtxt('EpisodeLogs/rate2corrected_s', delimiter=',')
    a = np.genfromtxt('EpisodeLogs/rate2corrected_ar', delimiter=',')[:, 0]

    s = s.reshape((s.shape[0], 4, 4)).astype(int)
    newS = np.zeros_like(s)

    for i in range(s.shape[0]):
        newS[i] = move(s[i], a[i])
    np.save('EpisodeLogs/newS', newS)
