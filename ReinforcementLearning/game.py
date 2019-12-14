'''A numpy-based 2048 game core implementation.'''

import numpy as np


class Game:

    def __init__(self, size=4, tile_to_win=np.inf, rate_2=0.9, random=False, enable_rewrite_board=False):
        '''

        :param size: the size of the board
        :param tile_to_win: the terminate tile to indicate `win`
        :param rate_2: the probability of the next element to be 2 (otherwise 4)
        :param random: a random initialized board (a harder mode)
        '''
        self.size = size
        self.tile_to_win = tile_to_win
        self.__rate_2 = rate_2
        self._score = 0
        if random:
            self.__board = \
                2 ** np.random.randint(1, 10, size=(self.size, self.size))
            self.__end = False
        else:
            self.__board = np.zeros((self.size, self.size), dtype=int)
            # initilize the board (with 2 entries)
            self._maybe_new_entry()
            self._maybe_new_entry()
        self.enable_rewrite_board = enable_rewrite_board
        assert not self.end

    def legalSuccessors(self):
        lM = []
        lS = []
        for direction in range(4):
            board_to_left = np.rot90(self.board, -direction)
            bCopy = np.copy(board_to_left)
            for row in range(self.size):
                core = self._merge(board_to_left[row])
                board_to_left[row, :len(core)] = core
                board_to_left[row, len(core):] = 0
            if not np.array_equal(board_to_left, bCopy):
                lM.append(direction)
                board_to_left = np.rot90(
                    board_to_left, direction).reshape((4, 4, 1))
                lS.append(board_to_left)
        return (lM, lS)

    def legalMoves(self):
        lM = []
        for direction in range(4):
            board_to_left = np.rot90(self.board, -direction)
            bCopy = np.array(board_to_left)
            for row in range(self.size):
                core = self._merge(board_to_left[row])
                board_to_left[row, :len(core)] = core
                board_to_left[row, len(core):] = 0
            if not np.array_equal(board_to_left, bCopy):
                lM.append(direction)
        return lM

    def customOrderLM(self):
        order = [0, 1, 3, 2]
        lM = []
        for direction in order:
            board_to_left = np.rot90(self.board, -direction)
            bCopy = np.array(board_to_left)
            for row in range(self.size):
                core = self._merge(board_to_left[row])
                board_to_left[row, :len(core)] = core
                board_to_left[row, len(core):] = 0
            if not np.array_equal(board_to_left, bCopy):
                lM.append(direction)
        return lM

    def moveWithoutSpawn(self, direction):
        board_to_left = np.rot90(self.board, -direction)
        for row in range(self.size):
            core = self._merge(board_to_left[row])
            board_to_left[row, :len(core)] = core
            board_to_left[row, len(core):] = 0

        # rotation to the original
        self.__board = np.rot90(board_to_left, direction)

    def move(self, direction):
        '''
        direction:
            0: left
            1: down
            2: right
            3: up
        '''
        # treat all direction as left (by rotation)
        board_to_left = np.rot90(self.board, -direction)
        for row in range(self.size):
            core = self._merge(board_to_left[row])
            board_to_left[row, :len(core)] = core
            board_to_left[row, len(core):] = 0

        # rotation to the original
        self.__board = np.rot90(board_to_left, direction)
        self._maybe_new_entry()

    def __str__(self):
        board = "State:"
        for row in self.board:
            board += ('\t' + '{:8d}' *
                      self.size + '\n').format(*map(int, row))
        board += "Max Tile: {0:d}".format(self.maxTile)
        return board

    @property
    def board(self):
        '''`NOTE`: Setting board by indexing,
        i.e. board[1,3]=2, will not raise error.'''
        return self.__board.copy()

    @board.setter
    def board(self, x):
        if self.enable_rewrite_board:
            assert self.__board.shape == x.shape
            self.__board = x.astype(self.__board.dtype)
        else:
            print("Disable to rewrite `board` manually.")

    @property
    def maxTile(self):
        return int(self.board.max())

    @property
    def score(self):
        return self._score

    @property
    def end(self):
        '''
        0: continue
        1: lose
        2: win
        '''
        if self.maxTile >= self.tile_to_win:
            return 2
        elif self.__end:
            return 1
        else:
            return 0

    def _maybe_new_entry(self):
        '''maybe set a new entry 2 / 4 according to `rate_2`'''
        where_empty = self._where_empty()
        if where_empty:
            selected = where_empty[np.random.randint(0, len(where_empty))]
            self.__board[selected] = \
                2 if np.random.random() < self.__rate_2 else 4
            self.__end = False
        else:
            self.__end = True

    def _where_empty(self):
        '''return where is empty in the board'''
        return list(zip(*np.where(self.board == 0)))

    def _merge(self, row):
        '''merge the row, there may be some improvement'''
        non_zero = row[row != 0]  # remove zeros
        core = [None]
        for elem in non_zero:
            if core[-1] is None:
                core[-1] = elem
            elif core[-1] == elem:
                core[-1] = 2 * elem
                self._score += core[-1]
                core.append(None)
            else:
                core.append(elem)
        if core[-1] is None:
            core.pop()
        return core
