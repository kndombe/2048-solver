import numpy as np
import os


class episodeLogger:

    def __init__(self, fname, diversify=True):
        if len(fname) == 0:
            raise Exception('File name for logger can\'t be empty')
        self.stateLog = open(fname + '_s', 'a')
        self.arLog = open(fname + '_ar', 'a')
        self.diversify = diversify

    def __close__(self):
        self.stateLog.close()
        self.arLog.close()

    def logTransition(self, s, a, r):
        if self.diversify:
            rot = np.random.randint(0, 4, dtype='int')
            s = np.rot90(s, rot)
            a = (a + rot) % 4
        self.stateLog.write(self.boardToFlattenedLoggedString(s) + '\n')
        self.arLog.write(str(a) + ',' + str(r) + '\n')

    def boardToFlattenedLoggedString(self, s):
        s[np.where(s == 0)] = 1
        s = np.log2(s).astype(int).flatten()
        return ','.join(map(str, s.tolist()))
