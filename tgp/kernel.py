from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, active_dims=None):

        self.active_dims = active_dims

    def slice_to_active(self, X1, X2):

        if self.active_dims is not None:
            X1 = X1[:, self.active_dims]
            X2 = X2[:, self.active_dims]

        assert(len(X1.shape) > 1 and len(X2.shape) > 1)

        return X1, X2

    @abstractmethod
    def calculate(self, X1, X2):
        """Compute the kernel.

        Args:
            X1 (np.array): An [N1 x D] matrix.
            X2 (np.array): An [N2 x D] matrix.

        Returns:
            np.array: An [N1 x N2] kernel matrix.
        """
        pass
