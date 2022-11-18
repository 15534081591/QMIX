"""onehot"""
import numpy as np


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        shape = list(tensor.shape)
        shape[3] = self.out_dim
        one_hot_label = np.zeros(shape, dtype=np.float32)
        for k in range(tensor.shape[1]):
            one_hot_label[0, k, :, :] = [[int(i == int(tensor[0][k][j][0])) for i in range(self.out_dim)] for j in
                                         range(tensor.shape[2])]
        return one_hot_label

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), np.float32
