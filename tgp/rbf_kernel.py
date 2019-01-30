from tgp.kernel import Kernel
from ml_tools.kernels import ard_rbf_kernel_efficient


class RBFKernel(Kernel):

    def __init__(self, lscales, sd):

        self.lscales = lscales
        self.sd = sd

    def calculate(self, X1, X2):

        return ard_rbf_kernel_efficient(X1, X2, self.sd, self.lscales)
