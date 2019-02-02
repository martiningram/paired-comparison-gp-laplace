from tgp.kernel import Kernel
from ml_tools.kernels import matern_kernel_32, matern_kernel_12


class MaternKernel32(Kernel):

    def __init__(self, lscales, sd, active_dims=None):

        super(MaternKernel32, self).__init__(active_dims=active_dims)

        self.lscales = lscales
        self.sd = sd

    def calculate(self, X1, X2):

        X1, X2 = self.slice_to_active(X1, X2)
        return matern_kernel_32(X1, X2, self.sd, self.lscales)


class MaternKernel12(Kernel):

    def __init__(self, lscales, sd, active_dims=None):

        super(MaternKernel12, self).__init__(active_dims=active_dims)

        self.lscales = lscales
        self.sd = sd

    def calculate(self, X1, X2):

        X1, X2 = self.slice_to_active(X1, X2)
        return matern_kernel_12(X1, X2, self.sd, self.lscales)
