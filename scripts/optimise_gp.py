import numpy as np
from tgp.gp_predictor import GPPredictor
from tgp.summed_kernel import SummedKernel
from tgp.rbf_kernel import RBFKernel
from scipy.optimize import minimize
from tdata.datasets.oncourt_dataset import OnCourtDataset


# Let's try 3 RBF kernel components.
dataset = OnCourtDataset()
df = dataset.get_stats_df()

one_year = df[df['year'] >= 2012]
one_year = one_year[one_year['year'] < 2018]

# Prepare the inputs
winners = one_year['winner'].values
losers = one_year['loser'].values
days_since_start = (one_year['start_date'] -
                    one_year['start_date'].min()).dt.days.values

n_matches = winners.shape[0]

lscales = np.array([14., 180., 720.]) / 300.
sds = np.array([0.1, 0.1, 0.1])

n_kerns = len(lscales)

theta_init = np.concatenate([lscales, sds], axis=0)


def to_minimise(theta):

    lscales = theta[:n_kerns]
    sds = theta[n_kerns:]

    print(lscales)
    print(sds)

    kernels = list()

    for cur_l, cur_sd in zip(lscales, sds):

        kernels.append(RBFKernel(np.array([cur_l]), cur_sd))

    kernel = SummedKernel(kernels)
    predictor = GPPredictor(kernel)
    predictor.fit(winners, losers, days_since_start)

    neg_marg_lik = -predictor.calculate_log_marg_lik()

    print(neg_marg_lik)

    return neg_marg_lik


result = minimize(to_minimise, theta_init, tol=1)

np.savez('result_vals', final_results=result.x)
