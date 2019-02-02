import numpy as np
from tgp.kernel import Kernel


class BiasKernel(Kernel):

    def __init__(self, sd):

        super(BiasKernel, self).__init__(active_dims=None)
        self.sd = sd

    def calculate(self, X1, X2):

        shape = (X1.shape[0], X2.shape[0])
        result = np.zeros(shape)

        return result + (self.sd ** 2)
