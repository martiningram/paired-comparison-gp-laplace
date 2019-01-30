from abc import ABC, abstractmethod


class Kernel(ABC):

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
