from tgp.kernel import Kernel
from tgp.kernel_functions import mlp_kernel


class MLPKernel(Kernel):

    def __init__(self, weight_sd, bias_sd, sd, active_dims=None):

        super(MLPKernel, self).__init__(active_dims=active_dims)

        self.weight_var = weight_sd**2
        self.bias_var = bias_sd**2
        self.var = sd**2

    def calculate(self, X1, X2):

        X1, X2 = self.slice_to_active(X1, X2)
        return mlp_kernel(X1, X2, self.var, self.weight_var, self.bias_var)
