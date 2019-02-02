from tgp.kernel import Kernel
from ml_tools.kernels import ard_rbf_kernel_efficient


class RBFKernel(Kernel):

    def __init__(self, lscales, sd, active_dims=None):

        super(RBFKernel, self).__init__(active_dims=active_dims)

        self.lscales = lscales
        self.sd = sd

    def calculate(self, X1, X2):

        X1, X2 = self.slice_to_active(X1, X2)

        return ard_rbf_kernel_efficient(X1, X2, self.sd, self.lscales)
