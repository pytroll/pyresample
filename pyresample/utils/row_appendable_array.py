import numpy as np


class RowAppendableArray:
    def __init__(self, reserve=0):
        self._reserve_size = reserve
        self._data = None
        self._cursor = 0

    def append_row(self, next_array):
        if self._data is None:
            self._data = np.empty((self._reserve_size, *next_array.shape[1:]), dtype=next_array.dtype)
        cursor_end = self._cursor + next_array.shape[0]
        if cursor_end > self._data.shape[0]:
            if len(next_array.shape) == 1:
                self._data = np.append(self._data, next_array)
            else:
                self._data = np.row_stack((self._data, next_array))
        else:
            self._data[self._cursor:cursor_end] = next_array
        self._cursor = cursor_end

    def to_array(self):
        return self._data[:self._cursor]
