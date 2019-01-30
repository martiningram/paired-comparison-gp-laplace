from tgp.kernel import Kernel
from ml_tools.kernels import brownian_kernel_1d


class BrownianKernel(Kernel):

    def __init__(self, sd):

        self.sd = sd

    def calculate(self, X1, X2):

        return brownian_kernel_1d(X1, X2, self.sd)
