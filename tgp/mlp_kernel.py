from tgp.kernel import Kernel
from ml_tools.kernels import mlp_kernel


class MLPKernel(Kernel):

    def __init__(self, weight_sd, bias_sd, sd):

        self.weight_var = weight_sd**2
        self.bias_var = bias_sd**2
        self.var = sd**2

    def calculate(self, X1, X2):

        return mlp_kernel(X1, X2, self.var, self.weight_var, self.bias_var)
