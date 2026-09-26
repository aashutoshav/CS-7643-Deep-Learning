import numpy as np


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        Hint: 1) You may implement the process with loops; 2) Name the output dimensions H_out and W_out
        """
        out = None
        ### TODO: BEGIN SOLUTION ###
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride

        H_out = int((H - k) / s + 1)
        W_out = int((W - k) / s + 1)
        out = np.zeros((N, C, H_out, W_out))

        for i in range(H_out):
            h_start = i * s
            h_end = h_start + k
            for j in range(W_out):
                w_start = j * s
                w_end = w_start + k
                window = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(window, axis=(2, 3))

        ### END SOLUTION ###
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout) -> None:
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        Hint: 1) You may implement the process with loops
              2) You may find np.unravel_index useful
        """
        x, H_out, W_out = self.cache
        ### TODO: BEGIN SOLUTION ###
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride

        dx = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    h_start = i * s
                    h_end = h_start + k
                    for j in range(W_out):
                        w_start = j * s
                        w_end = w_start + k
                        window = x[n, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        dx[n, c, h_start + max_idx[0], w_start + max_idx[1]] += dout[n, c, i, j]

        self.dx = dx
        ### END SOLUTION ###
