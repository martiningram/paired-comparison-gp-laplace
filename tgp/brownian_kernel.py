from tgp.kernel import Kernel
from tgp.kernel_functions import brownian_kernel_1d


class BrownianKernel(Kernel):

    def __init__(self, sd, active_dims=None):

        super(BrownianKernel, self).__init__(active_dims=active_dims)

        if self.active_dims is not None:
            assert(len(active_dims) == 1)

        self.sd = sd

    def calculate(self, X1, X2):

        X1, X2 = self.slice_to_active(X1, X2)
        return brownian_kernel_1d(X1, X2, self.sd)
