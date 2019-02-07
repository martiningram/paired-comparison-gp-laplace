from tgp.kernel import Kernel
from ml_tools.kernels import rq_kernel


class RationalQuadraticKernel(Kernel):

    def __init__(self, sd, lscales, alpha, active_dims=None):

        super(RationalQuadraticKernel, self).__init__(active_dims=active_dims)

        self.lscales = lscales
        self.sd = sd
        self.alpha = alpha

    def calculate(self, X1, X2):

        X1, X2 = self.slice_to_active(X1, X2)

        return rq_kernel(X1, X2, self.sd**2, self.lscales, self.alpha)
