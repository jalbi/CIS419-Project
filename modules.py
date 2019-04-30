import math
import torch
from torch.autograd import Variable, Function
import numpy as np


def dilate(x, dilation, init_dilation=1, pad_start=True):
    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = C_Pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self._in = 0
        self._out = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype

        self.num_pad = 0
        self.input_size = 0
        if data is None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self._in] = input.squeeze()
        self._in = (self._in + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        start = self._out - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self._out % dilation:self._out + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self._out + 1:dilation]

        self._out = (self._out + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self._in = 0
        self._out = 0


class CPad1d(Function):
    def __init__(self, target_size, dimension=0, value=0, pad_start=False):
        super(CPad1d, self).__init__()
        self.target_size = target_size
        self.dimension = dimension
        self.value = value
        self.pad_start = pad_start

    def forward(self, input):
        self.num_pad = self.target_size - input.size(self.dimension)
        assert self.num_pad >= 0, 'target size has to be greater than input size'

        self.input_size = input.size()

        size = list(input.size())
        size[self.dimension] = self.target_size
        output = input.new(*tuple(size)).fill_(self.value)
        output_copy = output

        # crop output
        if self.pad_start:
            output_copy = output_copy.narrow(self.dimension, self.num_pad, output_copy.size(self.dimension) - self.num_pad)
        else:
            output_copy = output_copy.narrow(self.dimension, 0, output_copy.size(self.dimension) - self.num_pad)

        output_copy.copy_(input)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new(*self.input_size).zero_()
        grad_output_copy = grad_output

        # crop grad_output
        if self.pad_start:
            grad_output_copy = grad_output_copy.narrow(self.dimension,
                                                       self.num_pad,
                                                       grad_output_copy.size(self.dimension) - self.num_pad)
        else:
            grad_output_copy = grad_output_copy.narrow(self.dimension,
                                                       0,
                                                       grad_output_copy.size(self.dimension) - self.num_pad)

        grad_input.copy_(grad_output_copy)
        return grad_input


def C_Pad_1d(input,
             target_size,
             dimension=0,
             value=0,
             pad_start=False):
    return CPad1d(target_size, dimension, value, pad_start)(input)
