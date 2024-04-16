import torch.cuda
# from isaacgymenvs.utils.torch_jit_utils import *
from isaacgym.torch_utils import *


class ResetBuffer():
    def __init__(self, n_dof, size=2048, device="cuda:0") -> None:
        self._buffer = torch.zeros((size, n_dof), dtype=torch.float32, device=device)
        self._buffer_size = size
        self._cur_buffer_size = 0
        self._fill_pointer = 0

    def __len__(self):
        return self._cur_buffer_size

    def save(self, data):
        _data_size = data.shape[0]
        if self._cur_buffer_size + _data_size <= self._buffer_size:
            self._buffer[self._cur_buffer_size:self._cur_buffer_size+_data_size] = data.clone()
            self._cur_buffer_size += _data_size
            self._fill_pointer = self._cur_buffer_size
        else:
            if self._fill_pointer + _data_size <= self._buffer_size:
                self._buffer[self._fill_pointer:self._fill_pointer+_data_size] = data.clone()
                self._fill_pointer = self._fill_pointer + _data_size
            else:
                _fill_from_begin_size = _data_size - (self._buffer_size - self._fill_pointer)
                self._buffer[self._fill_pointer:] = data[:self._buffer_size - self._fill_pointer].clone()
                self._buffer[:_fill_from_begin_size] = data[self._buffer_size - self._fill_pointer:].clone()
                self._fill_pointer = _fill_from_begin_size
            self._cur_buffer_size = min(self._buffer_size, self._cur_buffer_size+_data_size)

    def sample(self, num_samples):
        sample_indices = np.random.randint(self._cur_buffer_size, size=num_samples)
        return self._buffer[sample_indices].clone()
    