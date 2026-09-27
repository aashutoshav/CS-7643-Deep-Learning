import numpy as np


class Conv2D:
    """
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        Hint: 1) You may use np.pad for padding.
              2) You may implement the convolution with loops
        """
        out = None
        ### TODO: BEGIN SOLUTION ###
        N, C, H, W = x.shape
        p = self.padding
        s = self.stride
        k = self.kernel_size

        H_out = int((H + 2 * p - k) / s + 1)
        W_out = int((W + 2 * p - k) / s + 1)

        x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)
        out = np.zeros((N, self.out_channels, H_out, W_out))

        for i in range(H_out):
            h_start = i * s
            h_end = h_start + k
            for j in range(W_out):
                w_start = j * s
                w_end = w_start + k
                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]
                for f in range(self.out_channels):
                    out[:, f, i, j] = np.sum(x_slice * self.weight[f], axis=(1, 2, 3)) + self.bias[f]
        ### END SOLUTION ###
        self.cache = x
        return out

    def backward(self, dout) -> None:
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        Hint: 1) You may implement the convolution with loops
              2) don't forget padding when computing dx
        """
        x = self.cache
        ### TODO: BEGIN SOLUTION ###
        N, C, H, W = x.shape
        p = self.padding
        s = self.stride
        k = self.kernel_size
        _, F, H_out, W_out = dout.shape

        x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)
        dx_pad = np.zeros_like(x_pad)
        dw = np.zeros_like(self.weight)
        db = np.zeros_like(self.bias)

        db = np.sum(dout, axis=(0, 2, 3))

        for i in range(H_out):
            h_start = i * s
            h_end = h_start + k
            for j in range(W_out):
                w_start = j * s
                w_end = w_start + k

                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]

                for f in range(F):
                    dout_cur = dout[:, f, i, j][:, None, None, None]
                    dw[f] += np.sum(x_slice * dout_cur, axis=0)

                for n in range(N):
                    dout_n = dout[n, :, i, j][:, None, None, None]
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += np.sum(self.weight * dout_n, axis=0)

        if p > 0:
            self.dx = dx_pad[:, :, p:-p, p:-p]
        else:
            self.dx = dx_pad

        self.dw = dw
        self.db = db
        ### END SOLUTION ###
