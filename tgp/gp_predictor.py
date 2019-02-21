import autograd.numpy as np
import scipy.sparse as sps
from scipy.sparse import dok_matrix, block_diag
from autograd import elementwise_grad, jacobian
from autograd.scipy.stats import norm
from tgp.prepare_data import prepare_vectors
from sksparse.cholmod import cholesky
from tgp.kernel import Kernel
from autograd.scipy.misc import logsumexp


def is_sorted(l):

    return all(l[i] <= l[i+1] for i in range(len(l)-1))


def multivariate_normal_logpdf(x, cov):

    sign, logdet = np.linalg.slogdet(np.pi * 2 * cov)
    logdet = sign * logdet
    det_term = -0.5 * logdet

    # TODO: Could be improved by using some kind of Cholesky here rather than
    # inverse -- or even a solve.
    total_prior = det_term - 0.5 * x @ np.linalg.inv(cov) @ x

    return total_prior


class GPPredictor(object):

    def __init__(self, kernel):
        """
        Instantiates a new GPPredictor.

        Args:
            kernel (Kernel): The kernel to use.
        """

        self.is_fit = False

        self.kernel = kernel
        self.divide_by = 300.

        # Need to use Any here because of the lack of scipy support
        self.cached_big_inv = None
        self.likelihood_fun = GPPredictor.logit_log_lik

    @staticmethod
    def logit_log_lik(f_i, f_j):

        return -np.log1p(np.exp(-(f_i - f_j)))

    def fit(self, winners, losers, days_since_start, covariates=None):
        """
        Fits the GPPredictor.

        Args:
            winners (np.array): A numpy array of shape [N,] containing the
                winning players, where N is the number of matches.
            losers (np.array): A numpy array containing the losing players in
                the same shape as winners.
            days_since_start (np.array): A numpy array containing the days
                since the first match in the dataset, again in the same
                shape as the losers array.
            covariates (Optional[np.array]): If provided, additional covariates
                used to predict. These should be in shape [N, Nc], where N is
                the number of matches as above and Nc is the number of
                covariates.

        Returns:
            Nothing, but fits the GPPredictor, which can then be used to
            predict.
        """

        assert(is_sorted(days_since_start))
        assert not self.is_fit, 'GP Predictor should only be fit once!'

        self.encoder, self.s, self.e, self.w, self.l, self.X, cov_array = \
            prepare_vectors(winners, losers, days_since_start,
                            covariates=covariates)

        self.n_matches = self.w.shape[0] * 2

        self.prepare_x(cov_array)
        self.find_posterior_mode()

        self.is_fit = True

    def prepare_x(self, covariates=None):

        X_scaled = self.X / self.divide_by
        X_scaled = X_scaled[:, None]

        if covariates is not None:
            X_scaled = np.concatenate([X_scaled, covariates], axis=1)

        self.X = X_scaled

    def calculate_prior(self, f):

        total_logpdf = 0.

        for cur_start, cur_end in zip(self.s, self.e):

            cur_f = f[cur_start:cur_end]
            cur_X = self.X[cur_start:cur_end]
            cur_kernel = self.kernel.calculate(cur_X, cur_X)

            # Calculate the log pdf for this player
            cur_logpdf = multivariate_normal_logpdf(cur_f, cur_kernel)
            total_logpdf = total_logpdf + cur_logpdf

        return total_logpdf

    def calculate_likelihood(self, f):

        log_likelihood = np.sum(self.likelihood_fun(f[self.w], f[self.l]))

        return log_likelihood

    def calculate_posterior(self, f):

        cur_prior = self.calculate_prior(f)
        cur_likelihood = self.calculate_likelihood(f)
        return cur_prior + cur_likelihood

    @staticmethod
    def probit_log_lik(fi, fj):

        return norm.logcdf(fi - fj)

    def sparse_log_lik_hessian(self, f):

        f_shape = f.shape[0]
        big_hess = dok_matrix((f_shape, f_shape))

        grad_win_win = elementwise_grad(
            elementwise_grad(self.likelihood_fun, 0), 0)
        grad_lose_lose = elementwise_grad(
            elementwise_grad(self.likelihood_fun, 1), 1)
        big_hess[self.w, self.w] = grad_win_win(f[self.w], f[self.l])
        big_hess[self.l, self.l] = grad_lose_lose(f[self.w], f[self.l])
        grad_win_lose = elementwise_grad(
            elementwise_grad(self.likelihood_fun, 0), 1)
        big_hess[self.w, self.l] = grad_win_lose(f[self.w], f[self.l])
        big_hess[self.l, self.w] = grad_win_lose(f[self.w], f[self.l])

        return big_hess

    def sparse_prior_hessian(self, f):

        all_invs = list()

        for cur_start, cur_end in zip(self.s, self.e):

            cur_X = self.X[cur_start:cur_end]
            cur_kernel = self.kernel.calculate(cur_X, cur_X)
            cur_inv = np.linalg.inv(cur_kernel)

            all_invs.append(cur_inv)

        return -block_diag(all_invs)

    def posterior_hessian(self, f):

        lik_hess = self.sparse_log_lik_hessian(f)
        prior_hess = self.sparse_prior_hessian(f)

        return lik_hess + prior_hess

    def find_posterior_mode(self):

        # Initialise f
        f = np.zeros(self.n_matches)

        def to_minimize(f):

            cur_neg_value = -self.calculate_posterior(f)
            return cur_neg_value

        jac = jacobian(to_minimize)

        def hess(f):

            hessian = -self.posterior_hessian(f)
            return hessian

        # Write a Newton routine
        difference = 1.

        while difference > 1e-5:

            cur_hess = hess(f)
            cur_jac = jac(f)
            cur_val = to_minimize(f)

            # Multiply Hessian by value
            cur_chol = cholesky(cur_hess)

            sol = cur_chol.solve_A(cur_jac)

            new_f = f - sol
            difference = np.linalg.norm(f - new_f)

            f = new_f

        self.f_hat = f
        self.mode_chol = cur_chol

    def make_sparse_big_kernel(self):

        all_kerns = list()

        for cur_start, cur_end in zip(self.s, self.e):

            cur_X = self.X[cur_start:cur_end]
            cur_kernel = self.kernel.calculate(cur_X, cur_X)
            all_kerns.append(cur_kernel)

        return sps.block_diag(all_kerns)

    def make_sparse_inverse_kernel(self):

        all_kerns = list()

        for cur_start, cur_end in zip(self.s, self.e):

            cur_X = self.X[cur_start:cur_end]
            cur_kernel = self.kernel.calculate(cur_X, cur_X)
            all_kerns.append(np.linalg.inv(cur_kernel))

        return sps.block_diag(all_kerns)

    def calculate_log_marg_lik(self):
        """
        Calculates the log marginal likelihood.

        Returns:
            float: The log marginal likelihood of the data given the
            hyperparameters.
        """

        assert self.is_fit, ('GP Predictor must be fit before calculating '
                             'marginal log likelihood!')

        # TODO: I think I could just use the stored cholesky here...?

        big_inv_kern = self.make_sparse_inverse_kernel()
        sparse_hess = self.sparse_log_lik_hessian(self.f_hat)

        g_thth = big_inv_kern - sparse_hess

        g_chol = cholesky(g_thth.tocsc())
        logdet = g_chol.logdet()
        log_marg_lik = self.calculate_posterior(self.f_hat) - 0.5 * logdet

        return log_marg_lik

    def predict(self, player, days_since_start, covariates=None):
        """
        Predicts a player's skill on a day.

        Args:
            player (str): The player to predict.
            days_since_start (int): The day to predict, in days since the
                first match in the dataset used to fit the predictor.
            covariates (Optional[np.array]): If the predictor was fit with
                covariates, please provide the covariates to use for prediction
                here. This should be a row vector, i.e. [1, Nc], where Nc
                is the number of covariates.

        Returns:
            Tuple[float, float]: Tuple of mean and standard deviation predicted
            for the day given.
        """

        assert self.is_fit, ('GP Predictor must be fit before prediction is '
                             'possible!')

        if self.cached_big_inv is not None:
            big_inv_kern = self.cached_big_inv
        else:
            big_inv_kern = self.make_sparse_inverse_kernel()
            self.cached_big_inv = big_inv_kern

        x_star = np.array([days_since_start])
        x_star = x_star / self.divide_by
        x_star = x_star.reshape(-1, 1)

        if covariates is not None:
            x_star = np.concatenate([x_star, covariates], axis=1)

        all_kern = self.kernel.calculate(self.X, x_star)
        masked_kern = np.zeros_like(all_kern)

        if player in self.encoder.classes_:

            cur_index = self.encoder.transform([player])[0]
            cur_start = self.s[cur_index]
            cur_end = self.e[cur_index]

            # Zero out everything except these indices
            all_indices = np.arange(all_kern.shape[0])
            to_keep = np.arange(cur_start, cur_end)

            masked_kern[to_keep, :] = all_kern[to_keep, :]

        masked_kern = masked_kern.squeeze()
        pred_mean = np.matmul(masked_kern, (big_inv_kern * self.f_hat))

        k_star_star = self.kernel.calculate(x_star, x_star)
        inv_times_k_star = big_inv_kern @ masked_kern
        inv_term = masked_kern @ inv_times_k_star

        neg_post_chol = self.mode_chol

        solved_second_part = neg_post_chol.solve_A(inv_times_k_star)
        full_last_term = inv_times_k_star.T @ solved_second_part

        cur_mean, cur_sd = pred_mean, np.sqrt(full_last_term)

        return cur_mean, cur_sd
