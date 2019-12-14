'''
Augment transitions with 3 rotations
'''

import numpy as np

if __name__ == "__main__":
    s = np.load('EpisodeLogs/newS.npy')
    ar = np.genfromtxt('EpisodeLogs/rate2corrected_ar', delimiter=',')
    a = ar[:, 0]
    r = ar[:, 1]
    Q = np.load('Qvalues/initQ_complex.npy')

    numEx = a.size
    sAug = np.zeros((numEx*4, 4, 4))
    aAug = np.zeros(numEx*4)
    sAug[:numEx] = s
    aAug[:numEx] = a

    for dir in range(1, 4):
        sAug[numEx*dir: numEx*(dir+1)] = np.rot90(s, dir, axes=(1, 2))
        aAug[numEx*dir: numEx*(dir+1)] = (a + dir) % 4
    QAug = np.tile(Q, 4)
    rAug = np.tile(r, 4)
    np.save('EpisodeLogs/sAug', sAug)
    np.save('EpisodeLogs/aAug', aAug)
    np.save('EpisodeLogs/QAug', QAug)
    np.save('EpisodeLogs/rAug', rAug)
